"""Definition of the GPT Language Model largely copied from NanoGPT [1].

Some things have been removed or simplified since they're not necessary for inference.

[1]: https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    n_embd: int = 768
    dropout: float = 0.0
    # Bias in Linears and LayerNorms, like GPT-2.
    bias: bool = True


def new_gelu(x):
    """Gaussian Error Linear Unit (GELU) activation function.

    Copied from the nanogpt repo and identical to OpenAI GPT2 implementation. Paper:
    https://arxiv.org/abs/1606.08415
    """
    # fmt: off
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    # fmt: on


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


gpt_config = GPTConfig()
mlp = MLP(gpt_config).eval()

n_embd = gpt_config.n_embd


with open("models/124M/raw/model-h0-mlp-c_fc-w", "rb") as file_:
    tensor = np.frombuffer(file_.read(), dtype=np.float32)
    mlp.c_fc.weight.data = torch.tensor(tensor).reshape(4 * n_embd, n_embd)

with open("models/124M/raw/model-h0-mlp-c_fc-b", "rb") as file_:
    tensor = np.frombuffer(file_.read(), dtype=np.float32)
    mlp.c_fc.bias.data = torch.tensor(tensor).reshape(4 * n_embd)

with open("models/124M/raw/model-h0-mlp-c_proj-w", "rb") as file_:
    tensor = np.frombuffer(file_.read(), dtype=np.float32)
    mlp.c_proj.weight.data = torch.tensor(tensor).reshape(n_embd, 4 * n_embd)

with open("models/124M/raw/model-h0-mlp-c_proj-b", "rb") as file_:
    tensor = np.frombuffer(file_.read(), dtype=np.float32)
    mlp.c_proj.bias.data = torch.tensor(tensor).reshape(n_embd)

with open("models/124M/raw/model-wte", "rb") as file_:
    tensor = np.frombuffer(file_.read(), dtype=np.float32)
    inputs = torch.tensor(tensor).reshape(-1, 768)[:3]

outputs = mlp(inputs)

name_to_tensor = {
    "gpt_inputs": inputs,
    "gpt_outputs": outputs,
}

for name, tensor in name_to_tensor.items():
    if not os.path.exists(f"models/test/{name}"):
        with open(f"models/test/{name}", "wb") as file_:
            file_.write(tensor.reshape(-1).detach().numpy().tobytes())
