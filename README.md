# zig_gpt2
GPT-2 inference engine written in Zig. 

### Features:
* No third-party dependencies besides BLAS (Accelerate or OpenBLAS).
* No memory allocations at runtime.
* Can run [NanoGPT](https://github.com/karpathy/nanoGPT). 

### How to Run:

Download the GPT-2 checkpoint from OpenAI.
```bash
python3 download_weights.py
```

Build the Zig binary and run it with a prompt to generate completions:
```bash
zig build run -DOptimize=ReleaseFast
./zig-out/bin/zig_gpt2 "Marcus Aurelius said"
```

### How to Test:

Generate test data by forwarding random tensors through PyTorch ops.
```bash
python3 generate_test_data.py
```

Run tests. Verifies Zig ops produce the same output as PyTorch.
```bash
zig build test
```

---

### TODO

Implementation:
* ✅ Implement basic ops: Embedding, Linear, LayerNorm, GELU, Softmax, CausalSelfAttention.
* ✅ Implement transformer modules: MLP, Transformer block.
* ✅ Implement the full GPT model.
* ✅ Implement sampling from the model.
* ✅ Implement BPE encoding/decoding.
    
Efficiency:
* ✅ Replace custom linear algebra kernels with BLAS.
* ✅ Stream output as each new token is generated.
* ✅ Create central set of memory buffers and reuse them for each layer. No allocations at runtime.
* Add KV cache.
* Parallelize `softmax` and `gelu` operations.
