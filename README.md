# zig_inference
GPT2 inference engine written in Zig.

The north star is to run the [NanoGPT](https://github.com/karpathy/nanoGPT) model on CPU at 
least as fast as eager PyTorch.

Low probability of success.

---

TODOs:
* ✅ Implement basic ops: Embedding, Linear, LayerNorm, GELU, Softmax, CausalSelfAttention.
* ✅ Implement transformer modules: MLP, Transformer block.
* ✅ Implement the full GPT model.
* Implement sampling from the model.
* Implement token encoding / decoding.
* Create central set of buffers and reuse them for each layer, remove `allocators` from existing ops.

Wishlist:
* Replace `arrays` with `Vectors`.
* Replace custom linear algebra kernels with BLAS.
