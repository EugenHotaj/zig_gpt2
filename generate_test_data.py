import math
import os

import torch
import torch.nn.functional as F
from torch import nn

name_to_tensor = {}

# Generate Linear.
linear = nn.Linear(in_features=5, out_features=10)
inputs = torch.randn(3, 5)
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


inputs = torch.randn(3, 5)
outputs = gelu(inputs)
name_to_tensor.update({"gelu_inputs": inputs, "gelu_outputs": outputs})


# Generate softmax.
inputs = torch.randn(3, 5)
outputs = F.softmax(inputs, dim=-1)
name_to_tensor.update({"softmax_inputs": inputs, "softmax_outputs": outputs})


# Generate Embedding.
embedding = nn.Embedding(10, 5)
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
layer_norm = nn.LayerNorm(5)
inputs = torch.randn(3, 5)
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
batch_size, seq_len, n_head, head_dim = 3, 5, 3, 4
n_embed = n_head * head_dim

# Generate transpose intermediaries.
inputs = torch.randn(batch_size, seq_len, n_head, head_dim)
outputs = inputs.transpose(1, 2)
name_to_tensor.update({"transpose_inputs": inputs, "transpose_outputs": outputs})

# Generate causal self attention.
c_attn = nn.Linear(in_features=n_embed, out_features=3 * n_embed)
inputs = torch.randn(batch_size, seq_len, n_embed)
outputs = c_attn(inputs)

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

for name, tensor in name_to_tensor.items():
    if not os.path.exists(f"models/test/{name}"):
        with open(f"models/test/{name}", "wb") as file_:
            file_.write(tensor.reshape(-1).detach().numpy().tobytes())
