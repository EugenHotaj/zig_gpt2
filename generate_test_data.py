import math
import os

import torch
import torch.nn.functional as F
from torch import nn

name_to_tensor = {}

# Generate Linear.
linear = nn.Linear(in_features=768, out_features=4 * 768)
inputs = torch.randn(3, 768)
outputs = linear(inputs)
name_to_tensor.update(
    {
        "linear_weight": linear.weight,
        "linear_bias": linear.bias,
        "linear_inputs": inputs,
        "linear_outputs": outputs,
    }
)


# Generate GELU.
def gelu(x):
    """Gaussian Error Linear Unit (GELU) activation function.

    Copied from the nanogpt repo and identical to OpenAI GPT2 implementation. Paper:
    https://arxiv.org/abs/1606.08415
    """
    # fmt: off
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    # fmt: on


inputs = torch.randn(3, 768)
outputs = gelu(inputs)
name_to_tensor.update({"gelu_inputs": inputs, "gelu_outputs": outputs})


# Generate softmax.
inputs = torch.randn(3, 768)
outputs = F.softmax(inputs, dim=-1)
name_to_tensor.update({"softmax_inputs": inputs, "softmax_outputs": outputs})


# Generate Embedding.
embedding = nn.Embedding(10, 768)
inputs = torch.randint(0, 10, (3,))
outputs = embedding(inputs)
name_to_tensor.update(
    {
        "embedding_weight": embedding.weight,
        "embedding_inputs": inputs,
        "embedding_outputs": outputs,
    }
)


# Generate LayerNorm.
layer_norm = nn.LayerNorm(768)
inputs = torch.randn(3, 768)
outputs = layer_norm(inputs)
name_to_tensor.update(
    {
        "layer_norm_weight": layer_norm.weight,
        "layer_norm_bias": layer_norm.bias,
        "layer_norm_inputs": inputs,
        "layer_norm_outputs": outputs,
    }
)


# Generate causal self attention.
batch_size, seq_len, n_head, head_dim = 3, 5, 12, 64
n_embed = n_head * head_dim

# Generate transpose intermediaries.
inputs = torch.randn(batch_size, seq_len, n_head, head_dim)
outputs = inputs.transpose(1, 2)
name_to_tensor.update({"transpose_inputs": inputs, "transpose_outputs": outputs})

# Generate split intermediaries.
inputs = torch.randn(batch_size, seq_len, 3 * n_embed)
q, k, v = inputs.split(n_embed, dim=2)
name_to_tensor.update(
    {"split_inputs": inputs, "split_q": q, "split_k": k, "split_v": v}
)


inputs = torch.randn(batch_size, seq_len, n_embed)
c_attn = nn.Linear(in_features=n_embed, out_features=3 * n_embed)
outputs = c_attn(inputs)
name_to_tensor.update(
    {
        "attn_inputs": inputs,
        "attn_c_attn_weight": c_attn.weight,
        "attn_c_attn_bias": c_attn.bias,
    }
)

# Generate intermediaries from scaled dot product attention.
q, k, v = outputs.split(n_embed, dim=2)
q = q.view(batch_size, seq_len, n_head, n_embed // n_head).transpose(1, 2)
k = k.view(batch_size, seq_len, n_head, n_embed // n_head).transpose(1, 2)
v = v.view(batch_size, seq_len, n_head, n_embed // n_head).transpose(1, 2)
mask = torch.tril(torch.ones(5, 5).view(1, 1, 5, 5))
attn = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
attn = attn.masked_fill(mask[:, :, :5, :5] == 0, float("-inf"))
attn = F.softmax(attn, dim=-1)
outputs = attn @ v
name_to_tensor.update({"sdpa_q": q, "sdpa_k": k, "sdpa_v": v, "sdpa_outputs": outputs})

inputs = outputs.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embed)
c_proj = nn.Linear(n_embed, n_embed)
outputs = c_proj(inputs)
name_to_tensor.update(
    {
        "attn_c_proj_weight": c_proj.weight,
        "attn_c_proj_bias": c_proj.bias,
        "attn_outputs": outputs,
    }
)

for name, tensor in name_to_tensor.items():
    if not os.path.exists(f"models/test/{name}"):
        with open(f"models/test/{name}", "wb") as file_:
            file_.write(tensor.reshape(-1).detach().numpy().tobytes())
