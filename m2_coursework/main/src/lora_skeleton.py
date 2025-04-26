import math
import os, sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from accelerate import Accelerator
from preprocessor import load_and_preprocess
from predict import load_qwen, load_lora, LoRALinear
import argparse

parser = argparse.ArgumentParser(description="Train LoRA-augmented Qwen model.")
parser.add_argument("--lora_rank", type=int, default=4, help="LoRA rank (r)")
parser.add_argument("--learning_rate", type=str, default="1e-5", help="Learning rate")
parser.add_argument("--max_steps", type=int, default=100, help="Number of training steps")
parser.add_argument("--context_length", type=int, default=512, help="Maximum context length")
parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
parser.add_argument("--resume_training", action="store_true", help="Resume training from checkpoint")
parser.add_argument("--inherit_exp", type=str, help="exp_name of the model to inherit from")

args = parser.parse_args()

lora_rank = args.lora_rank
batch_size = args.batch_size
learning_rate = args.learning_rate
max_steps = args.max_steps
max_ctx_length = args.context_length

exp_name = "_".join(["lora", "r"+str(lora_rank), "lr"+str(learning_rate), 
          "s"+str(max_steps), "cl"+str(max_ctx_length)])

learning_rate = float(learning_rate)

if exp_name != "original":
    if args.resume_training:
        model, tokenizer = load_lora(args.inherit_exp)
    else:
        model, tokenizer = load_lora(exp_name)
else:
    model, tokenizer = load_qwen()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Process the data into sequences of text
train_texts, val_texts = load_and_preprocess("lotka_volterra_data.h5")


# Modified tokenization with chunking
def process_sequences(texts, tokenizer, max_length=512, stride=256):
    all_input_ids = []
    for text in texts:
        # Apply Qwen's tokenization scheme to the text:
        encoding = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        seq_ids = encoding.input_ids[0]

        # Create sliding windows to further divide the data into chunks:
        for i in range(0, len(seq_ids), stride):
            chunk = seq_ids[i : i + max_length]
            if len(chunk) < max_length:
                chunk = torch.cat(
                    [
                        chunk,
                        torch.full((max_length - len(chunk),), tokenizer.pad_token_id),
                    ]
                )
            all_input_ids.append(chunk)
    return torch.stack(all_input_ids)


train_input_ids = process_sequences(
    train_texts, tokenizer, max_ctx_length, stride=max_ctx_length // 2
)
val_input_ids = process_sequences(
    val_texts, tokenizer, max_ctx_length, stride=max_ctx_length
)



optimizer = torch.optim.Adam(
    (p for p in model.parameters() if p.requires_grad), lr=learning_rate
)
train_dataset = TensorDataset(train_input_ids)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Prepare components with Accelerator
accelerator = Accelerator()
model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

def evaluate_loss(model, dataloader, accelerator):
    model.eval()
    losses = []
    with torch.no_grad():
        for (batch,) in dataloader:
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            losses.append(accelerator.gather(loss))
    model.train()
    return torch.stack(losses).cpu().mean().item()

# Prepare test/validation data once
val_dataset = TensorDataset(val_input_ids)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
val_loader = accelerator.prepare(val_loader)

model.train()
steps = 0
eval_every = 20
best_val_loss = float("inf")

while steps < max_steps:
    progress_bar = tqdm(train_loader, desc=f"Steps {steps}")
    for (batch,) in progress_bar:
        optimizer.zero_grad()
        outputs = model(batch, labels=batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        steps += 1
        
        progress_bar.set_postfix(loss=loss.item())

        if steps % eval_every == 0 or steps == 1:
            val_loss = evaluate_loss(model, val_loader, accelerator)

            print(f"\n[Step {steps}] Validation Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_to_save = accelerator.unwrap_model(model)
                # Save only trainable parameters (i.e., LoRA weights)
                lora_state_dict = {
                    k: v for k, v in model_to_save.state_dict().items()
                    if any(x in k for x in ['q_proj.A', 'q_proj.B', 'v_proj.A', 'v_proj.B', 'k_proj.A', 'k_proj.B'])
                }
                torch.save(lora_state_dict, exp_name+".pt")
                print(f"[Step {steps}] Saved new best model (val_loss={val_loss:.4f})")

        if steps >= max_steps:
            break
        print(f"final best loss = {val_loss:.4f}")
