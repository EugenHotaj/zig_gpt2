"""Definition of GPT-2, largely copied from NanoGPT [1].

NOTE: There is some divergence from NanoGPT: 
    * Always use biases and the default LayerNorm (like GPT-2).
    * Use the same vocab size as GPT-2.
    * Remove dropout (does not affect inference).
    * Stripped down GPT module which only includes forward and returns logits.
    * No support for PyTorch 2.0 / flash attention.

[1]: https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

import math
import os
from dataclasses import dataclass

import numpy as np
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
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


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # https://paperswithcode.com/method/weight-tying
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx):
        _, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)  # (t)

        # Forward the GPT model.
        tok_emb = self.transformer.wte(idx)  # token embeddings (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings (T, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # TODO(eugenhotaj): inference-time mini-optimization: only forward the lm_head
        # on the very last position
        # logits = self.lm_head(x[:, [-1], :]) # Using [-1] to preserve the time dim.
        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(self, idx, new_tokens, temp=0.8):
        assert len(idx) + new_tokens <= self.config.block_size

        for _ in range(new_tokens):
            logits = self(idx)[:, -1, :] / temp
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def load_linear(module: nn.Linear, name: str, in_f: int, out_f: int) -> None:
    with open(f"models/124M/raw/model-{name}-w", "rb") as file_:
        tensor = np.frombuffer(file_.read(), dtype=np.float32)
        module.weight.data = torch.tensor(tensor).reshape(out_f, in_f)

    with open(f"models/124M/raw/model-{name}-b", "rb") as file_:
        tensor = np.frombuffer(file_.read(), dtype=np.float32)
        module.bias.data = torch.tensor(tensor).reshape(out_f)


def load_layernorm(module: nn.LayerNorm, name: str) -> None:
    with open(f"models/124M/raw/model-{name}-g", "rb") as file_:
        tensor = np.frombuffer(file_.read(), dtype=np.float32)
        module.weight.data = torch.tensor(tensor)

    with open(f"models/124M/raw/model-{name}-b", "rb") as file_:
        tensor = np.frombuffer(file_.read(), dtype=np.float32)
        module.bias.data = torch.tensor(tensor)


def load_attention(module: CausalSelfAttention, layer: int, n_embd: int) -> None:
    load_linear(module.c_attn, f"h{layer}-attn-c_attn", n_embd, 3 * n_embd)
    load_linear(module.c_proj, f"h{layer}-attn-c_proj", n_embd, n_embd)


def load_mlp(module: MLP, layer: int, n_embd: int) -> None:
    load_linear(module.c_fc, f"h{layer}-mlp-c_fc", n_embd, 4 * n_embd)
    load_linear(module.c_proj, f"h{layer}-mlp-c_proj", 4 * n_embd, n_embd)


def load_block(module: Block, layer: int, n_embd: int) -> None:
    load_layernorm(module.ln_1, f"h{layer}-ln_1")
    load_attention(module.attn, layer, n_embd)
    load_layernorm(module.ln_2, f"h{layer}-ln_2")
    load_mlp(module.mlp, layer, n_embd)


def load_embedding(
    module: nn.Embedding, name: str, vocab_size: int, n_embd: int
) -> None:
    with open(f"models/124M/raw/model-{name}", "rb") as file_:
        tensor = np.frombuffer(file_.read(), dtype=np.float32)
        tensor = torch.tensor(tensor).reshape(vocab_size, n_embd)
        module.weight.data = tensor


def load_gpt(module: GPT, config: GPTConfig) -> None:
    load_embedding(module.transformer.wte, "wte", config.vocab_size, config.n_embd)
    load_embedding(module.transformer.wpe, "wpe", config.block_size, config.n_embd)
    for i in range(config.n_layer):
        load_block(module.transformer.h[i], i, config.n_embd)
    load_layernorm(module.transformer.ln_f, "ln_f")
    # Loading wte should automatically load lm_head since they point to the same tensor.
    assert module.lm_head.weight is module.transformer.wte.weight


gpt_config = GPTConfig()
gpt = GPT(gpt_config).eval()
load_gpt(gpt, gpt_config)

encoder = tiktoken.get_encoding("gpt2")
encoded = encoder.encode(
    "Marcus Aurelius said thus: ", allowed_special={"<|endoftext|>"}
)
inputs = torch.tensor(encoded).view((1, -1))

outputs = gpt.generate(inputs, 10)
outputs = encoder.decode(outputs.tolist()[0])
print(outputs)
# fmt: off
# Zig outputs:
print(encoder.decode([35110, 43737, 75, 3754, 531, 4145, 25, 220, 1849, 5246, 14931, 314, 14960, 616, 29644, 357, 34470, 5106]))
# fmt: on

name_to_tensor = {
    "gpt_inputs": inputs,
}

for name, tensor in name_to_tensor.items():
    if not os.path.exists(f"models/test/{name}"):
        with open(f"models/test/{name}", "wb") as file_:
            file_.write(tensor.reshape(-1).detach().numpy().tobytes())
