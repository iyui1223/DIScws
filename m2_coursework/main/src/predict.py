import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from preprocessor import PrePostProcessor, read_hdf5
import torch.nn as nn


def load_qwen():
    """Loads the Qwen model and tokenizer."""
    from transformers import AutoModelForCausalLM
    from transformers import AutoTokenizer

    model_name="Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    # Freeze all parameters except LM head bias
    for param in model.parameters():
        param.requires_grad = False

    # Add trainable bias to logits
    assert model.lm_head.bias is None
    model.lm_head.bias = torch.nn.Parameter(
        torch.zeros(model.config.vocab_size, device=model.device)
    )
    model.lm_head.bias.requires_grad = True

    return model, tokenizer


class LoRALinear(nn.Module):
    
    def __init__(self, original_linear: nn.Linear, r: int, alpha: int = None):
        super().__init__()
        assert isinstance(original_linear, nn.Linear)
        self.original_linear = original_linear
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False
        in_dim = original_linear.in_features
        out_dim = original_linear.out_features
        self.r = r
        self.alpha = alpha if alpha else r

        device = original_linear.weight.device
        self.A = nn.Parameter(torch.empty(r, in_dim, device=device))
        self.B = nn.Parameter(torch.zeros(out_dim, r, device=device))
        
        # Initialise A with He initialization
        nn.init.kaiming_normal_(self.A, nonlinearity="linear")

    def forward(self, x):
        base_out = self.original_linear(x)
        lora_out = (x @ self.A.T) @ self.B.T
        return base_out + lora_out * (self.alpha / self.r)
    

def load_lora(exp_name):
    """
    Load Qwen model with LoRA adapters and load trained LoRA weights.
    Assumes LoRA was applied to q_proj and v_proj only.
    """
    import re
    from qwen import load_qwen  # Or just call the function defined above

    # Parse hyperparameters from exp_name string
    match = re.match(r"lora_r(\d+)_lr([\de\.-]+)_s(\d+)_cl(\d+)", exp_name)
    if not match:
        raise ValueError(f"Could not parse hyperparameters from exp_name: {exp_name}")

    r = int(match.group(1))  # LoRA rank

    # Load base model and tokenizer
    model, tokenizer = load_qwen()

    # Reapply LoRA to q_proj and v_proj
    for layer in model.model.layers:
        layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=r)
        layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=r)

    # Load LoRA weights
    lora_path = f"{exp_name}.pt"
    if not os.path.exists(lora_path):
        print(f"LoRA weights not found in {lora_path} ")
    else:
        print(f"LoRA weights found in {lora_path} -- loading resume training")
        state_dict = torch.load(lora_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    return model, tokenizer


if __name__ == "__main__":
    infile = "lotka_volterra_data.h5"
    parser = argparse.ArgumentParser(description="Run predictions with Qwen model")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    args = parser.parse_args()
    
    prediction_dir = "./predictions/"+args.exp_name

    time, trajectories = read_hdf5(infile)
    processor = PrePostProcessor()

    # Define dimensions
    systems, tsteps, types = trajectories.shape
    in_steps = tsteps // 2  # First half as input
    
    predicted_steps = tsteps - in_steps  # Remaining steps to predict

    # Load model
    if args.exp_name != "original":
        model, tokenizer = load_lora(args.exp_name)
    else:
        model, tokenizer = load_qwen()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    split_index = int(systems * 0.7)

    # Process each system independently
    for i in range(split_index, split_index+30): # systems): # validation set
        prediction_file = os.path.join(prediction_dir, f"system{i}.npy")
        prediction_text_file = os.path.join(prediction_dir, f"system{i}.txt")
        if os.path.exists(prediction_file) or os.path.exists(prediction_text_file):
            print(f"Skipping system {i}, prediction already exists.")
        else:
            print(f"Processing system {i}...")
            tokenized = processor.preprocess(time, trajectories[i:i+1, :in_steps, :])
            
            if isinstance(tokenized, list):
                tokenized = torch.tensor(tokenized, dtype=torch.long)

            tokenized_input = tokenized.to(device)  # Add batch dim
            tokenized_input = tokenized_input.to(torch.long)
            generated_ids = model.generate(
                tokenized_input, 
                max_new_tokens=predicted_steps * 8,  # 8 tokens per timestep
                do_sample=True
            )
            generated_ids = generated_ids[:, len(tokenized_input[0]):]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            prediction = processor.postprocess(response)
            if isinstance(prediction, str):
                f = open(prediction_file.replace("npy", "txt"), 'w')
                f.write(prediction)
                f.close()
            else:
                np.save(prediction_file, prediction)
            print(f"Prediction for system {i} saved to {prediction_file}.")
