"""Definition of GPT-2, largely copied from NanoGPT [1].

NOTE: There is some divergence from NanoGPT, such as always including biases (since 
GPT-2) uses them, using the default LayerNorm, not supporting flash attention, etc.

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
    block_size: int = 1024
    n_head: int = 12
    n_embd: int = 768


def new_gelu(x):
    """Gaussian Error Linear Unit (GELU) activation function.

    Copied from the nanogpt repo and identical to OpenAI GPT2 implementation. Paper:
    https://arxiv.org/abs/1606.08415
    """
    # fmt: off
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    # fmt: on


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Key, query, value projections for all heads, but in a batch.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Causal mask.
        bias = torch.tril(torch.ones(config.block_size, config.block_size))
        bias = bias.view(1, 1, config.block_size, config.block_size)
        self.register_buffer("bias", bias)

    def forward(self, x):
        # Batch size, sequence length, embedding dimensionality (n_embd).
        B, T, C = x.size()
        hs = C // self.n_head

        # Calculate query, key, values for all heads in batch and move head forward to
        # be the batch dim.
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)

        # Manual implementation of attention.
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Re-assemble all head outputs side by side.
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection.
        y = self.c_proj(y)
        return y


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


def load_linear(module: nn.Linear, layer_name: str, in_f: int, out_f: int) -> None:
    with open(f"models/124M/raw/model-{layer_name}-w", "rb") as file_:
        tensor = np.frombuffer(file_.read(), dtype=np.float32)
        module.weight.data = torch.tensor(tensor).reshape(out_f, in_f)

    with open(f"models/124M/raw/model-{layer_name}-b", "rb") as file_:
        tensor = np.frombuffer(file_.read(), dtype=np.float32)
        module.bias.data = torch.tensor(tensor).reshape(out_f)


def load_layernorm(module: nn.LayerNorm, layer: int, idx: int) -> None:
    with open(f"models/124M/raw/model-h{layer}-ln_{idx}-b", "rb") as file_:
        tensor = np.frombuffer(file_.read(), dtype=np.float32)
        module.weight.data = torch.tensor(tensor)

    with open(f"models/124M/raw/model-h{layer}-ln_{idx}-g", "rb") as file_:
        tensor = np.frombuffer(file_.read(), dtype=np.float32)
        module.bias.data = torch.tensor(tensor)


def load_attention(module: CausalSelfAttention, layer: int, n_embd: int) -> int:
    load_linear(module.c_attn, f"h{layer}-attn-c_attn", n_embd, 3 * n_embd)
    load_linear(module.c_proj, f"h{layer}-attn-c_proj", n_embd, n_embd)


def load_mlp(module: MLP, layer: int, n_embd: int) -> None:
    load_linear(module.c_fc, f"h{layer}-mlp-c_fc", n_embd, 4 * n_embd)
    load_linear(module.c_proj, f"h{layer}-mlp-c_proj", 4 * n_embd, n_embd)


def load_block(module: Block, layer: int, n_embd: int) -> None:
    load_layernorm(module.ln_1, layer, 1)
    load_attention(module.attn, layer, n_embd)
    load_layernorm(module.ln_2, layer, 2)
    load_mlp(module.mlp, layer, n_embd)


gpt_config = GPTConfig()
with open(f"models/124M/raw/model-wte", "rb") as file_:
    tensor = np.frombuffer(file_.read(), dtype=np.float32)
    tensor = torch.tensor(tensor).reshape(-1, 768)
    inputs = torch.randint(0, tensor.shape[0], (15,))
    inputs = tensor[inputs].reshape(3, 5, 768)


block = Block(gpt_config)
load_block(block, 0, gpt_config.n_embd)
block = block.eval()
outputs = block(inputs)

name_to_tensor = {
    "gpt_inputs": inputs,
    "gpt_outputs": outputs,
}

for name, tensor in name_to_tensor.items():
    if not os.path.exists(f"models/test/{name}"):
        with open(f"models/test/{name}", "wb") as file_:
            file_.write(tensor.reshape(-1).detach().numpy().tobytes())
