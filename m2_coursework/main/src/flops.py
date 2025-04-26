import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

ops_summary = []

def compute_ops(layer_type, in_shape, out_shape, has_bias=False):
    """Compute estimated multiplications and summations for each layer type."""
    mults = sums = others = 0
    batch, seq_len = in_shape[0], in_shape[1]
    
    if layer_type == "Linear":
        in_features = in_shape[-1]
        out_features = out_shape[-1]
        mults = in_features * out_features
        sums = (in_features - 1) * out_features
        if has_bias:
            sums += (out_features)

    elif layer_type == "SiLU": # silu(x)=x∗σ(x),where σ(x) is the logistic sigmoid.
        out_features = out_shape[-1]
        others +=  batch * seq_len * (10 + 1 + 1) # for calculation of exp and division
        sums = mults = batch * seq_len * out_features

    elif "Norm" in layer_type: # mean, power, rsqrt, a sum, weight multiplication
        # RMSNorm
        features = in_shape[-1]
        sums += features - 1 + 1 # sum for mean and of epsiron
        mults += features * 3 # for pow, sqrt, and weight multiply each
        others += (1 + 10) # for division in mean and rsqrt each 

    return batch * seq_len * mults, batch * seq_len * sums, batch * seq_len * others


def register_hooks(model):
    def hook_fn(module, input, output):
        name = module.__class__.__name__
        if isinstance(input, (tuple, list)):
            input = input[0]
        if not hasattr(output, "shape"):
            return

        in_shape = tuple(input.shape)
        out_shape = tuple(output.shape)

        has_bias = hasattr(module, "bias") and module.bias is not None

        mults, sums, others = compute_ops(name, in_shape, out_shape, has_bias=has_bias)
        ops_summary.append({
            "type": name,
            "in": in_shape,
            "out": out_shape,
            "mults": mults,
            "sums": sums,
            "others": others,
        })

    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.SiLU, nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)) \
           or "RMSNorm" in module.__class__.__name__:
            module.register_forward_hook(hook_fn)


def run_analysis():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    print(model)

    register_hooks(model)

    # Dummy input
#    dummy = tokenizer("D"*512*8, return_tensors="pt").to(device) #
    dummy = tokenizer("D", return_tensors="pt").to(device) # produce result for a single token

    with torch.no_grad(): # backword propagation not needed
#        _ = model(**dummy, max_new_tokens=512*8)
        _ = model(**dummy, max_new_tokens=1) # returns double the length of input tokens

    total_mults = sum(op["mults"] for op in ops_summary)
    total_sums = sum(op["sums"] for op in ops_summary)
    total_others = sum(op["others"] for op in ops_summary)

    print(f"{'Layer':15s} {'Input':20s} {'Output':20s} {'Mults':>12s} {'Sums':>12s} {'Others':>12s}")
    print("-" * 80)
    for op in ops_summary:
        print(f"{op['type']:15s} {str(op['in']):20s} {str(op['out']):20s} {op['mults']:12,d} {op['sums']:12,d} {op['others']:12,d}")

    print("-" * 80)
    print(f"{'Total':>58s}: {total_mults:,} multiplications, {total_sums:,} summations, and {total_others:,} other operations")
    print(f"Single forward pass accounts {total_mults+total_sums+total_others} x token_length x batch_size flops")

if __name__ == "__main__":
    run_analysis()

