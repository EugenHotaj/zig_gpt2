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

q = torch.randn(2, 3, 5, 4)
k = torch.randn(2, 3, 5, 4)
qk = q @ k.transpose(-2, -1)

name_to_tensor.update(
    {
        "attn_q": q,
        "attn_k": k,
        "attn_qk": qk,
    }
)


for name, tensor in name_to_tensor.items():
    if not os.path.exists(f"models/test/{name}"):
        with open(f"models/test/{name}", "wb") as file_:
            file_.write(tensor.reshape(-1).detach().numpy().tobytes())
