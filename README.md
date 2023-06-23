# zig_inference
GPT2 inference engine written in Zig. 

The inference engine can run [NanoGPT](https://github.com/karpathy/nanoGPT) **~2x faster than eager PyTorch** (at least on my CPU). 

### How to Run:
Download the GPT-2 weights from OpenAI, load them in PyTorch, and generate some test data by forwarding 
a few examples:
```bash
python3 download_weights.py
time python3 generate_nano_gpt.py
```

Build the zig binary and run it on the generated data:
```bash
zig build-exe ./src/main.zig -O ReleaseFast -fstrip -fsingle-threaded -target x86_64-macos
time ./main
```
The zig binary loads the GPT-2 weights, forwards the same examples, and checks that the output matches PyTorch's.

---

TODOs:
* ✅ Implement basic ops: Embedding, Linear, LayerNorm, GELU, Softmax, CausalSelfAttention.
* ✅ Implement transformer modules: MLP, Transformer block.
* ✅ Implement the full GPT model.
* Implement sampling from the model.
* Implement token encoding / decoding.
* ✅ Create central set of buffers and reuse them for each layer, remove `allocators` from existing ops.

Wishlist:
* Replace `arrays` with `Vectors`.
* Replace custom linear algebra kernels with BLAS.
