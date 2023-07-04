# zig_gpt2
GPT-2 inference engine written in Zig. 

### Features:
* Only third-party dependency is OpenBLAS.
* No memory allocations at runtime.
* Can run [NanoGPT](https://github.com/karpathy/nanoGPT). 

### How to Run:

*NOTE:* `zig_gpt2` uses OpenBLAS under the hood for fast matrix multiplications. To run, you first need to build OpenBLAS and
place it in `lib/OpenBLAS`. This should be as easy as cloning the [OpenBLAS GitHub repo](https://github.com/xianyi/OpenBLAS)
and running `make`.

Download the GPT-2 weights from OpenAI, load them in PyTorch, and generate some test data by forwarding 
a few examples:
```bash
python3 download_weights.py
time python3 generate_nano_gpt.py
```

Build the zig binary and run it on the generated data:
```bash
zig build run -DOptimize=ReleaseFast
```
The zig binary loads the GPT-2 weights, forwards the same examples, and checks that the output matches PyTorch's. It also
generates a bunch of tokens.

---

TODOs:
* ✅ Implement basic ops: Embedding, Linear, LayerNorm, GELU, Softmax, CausalSelfAttention.
* ✅ Implement transformer modules: MLP, Transformer block.
* ✅ Implement the full GPT model.
* ✅ Implement sampling from the model.
* Implement token encoding / decoding.
* ✅ Create central set of buffers and reuse them for each layer, remove `allocators` from existing ops.
* ✅ Parallelize Linear and CausalSelfAttention operations. (Replaced with BLAS instead.)
* ✅ Replace custom linear algebra kernels with BLAS.
* Parallelize softmax and gelu operations.