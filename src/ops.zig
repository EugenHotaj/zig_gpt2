const std = @import("std");
const blas = @cImport(@cInclude("cblas.h"));

pub const Linear = struct {
    const Self = @This();

    in_features: usize,
    out_features: usize,
    weight: []const f32, // Weights must be provided in *column major* order!
    bias: []const f32,

    // Fields below are private and should not be set directly.
    use_bias: bool,

    pub fn init(in_features: usize, out_features: usize, weight: []const f32, bias: []const f32) Self {
        return Self{
            .in_features = in_features,
            .out_features = out_features,
            .use_bias = true,
            .weight = weight,
            .bias = bias,
        };
    }

    pub fn init_no_bias(in_features: usize, out_features: usize, weight: []const f32) Self {
        return Self{
            .in_features = in_features,
            .out_features = out_features,
            .use_bias = false,
            .weight = weight,
            .bias = undefined,
        };
    }

    pub fn forward(self: Self, inputs: []const f32, outputs: []f32) void {
        const batch_size = inputs.len / self.in_features;
        var beta: f32 = 0.0;
        if (self.use_bias) {
            for (0..batch_size) |b| {
                @memcpy(outputs[b * self.out_features .. (b + 1) * self.out_features], self.bias);
            }
            beta = 1.0;
        }
        blas.cblas_sgemm(
            blas.CblasRowMajor,
            blas.CblasNoTrans,
            blas.CblasTrans,
            @intCast(i32, batch_size),
            @intCast(i32, self.out_features),
            @intCast(i32, self.in_features),
            1.0,
            inputs.ptr,
            @intCast(i32, self.in_features),
            self.weight.ptr,
            @intCast(i32, self.in_features),
            beta,
            outputs.ptr,
            @intCast(i32, self.out_features),
        );
    }
};

pub const Embedding = struct {
    const Self = @This();

    emb_dim: usize,
    weight: []const f32,

    pub fn init(emb_dim: usize, weight: []const f32) Self {
        return Self{ .emb_dim = emb_dim, .weight = weight };
    }

    pub fn forward(self: Self, idxs: []const usize, embeddings: []f32) void {
        for (0..idxs.len) |i| {
            const idx = idxs[i];
            // TODO(eugenhotaj): There is no reason to copy memory here. We should
            // instead return views (pointers) to the embeddings.
            std.mem.copyForwards(
                f32,
                embeddings[i * self.emb_dim .. (i + 1) * self.emb_dim],
                self.weight[self.emb_dim * idx .. self.emb_dim * (idx + 1)],
            );
        }
    }
};

pub const LayerNorm = struct {
    const Self = @This();

    n_features: usize,
    weight: []const f32,
    bias: []const f32,
    eps: f32 = 1e-5,

    pub fn init(n_features: usize, weight: []const f32, bias: []const f32) Self {
        return Self{ .n_features = n_features, .weight = weight, .bias = bias };
    }

    pub fn forward(self: Self, inputs: []f32) void {
        const batch_size = inputs.len / self.n_features;
        for (0..batch_size) |b| {
            // Compute the mean and variance.
            var mean: f64 = 0.0;
            var std_: f64 = 0.0;
            for (0..self.n_features) |i| {
                const x = inputs[b * self.n_features + i];
                mean += x;
                std_ += x * x;
            }
            const n = @intToFloat(f64, self.n_features);
            mean /= n;
            std_ = @sqrt((std_ / n) - (mean * mean) + self.eps);

            // Normalize.
            for (0..self.n_features) |i| {
                const idx = b * self.n_features + i;
                const x = inputs[idx];
                const result = (x - mean) / std_ * self.weight[i] + self.bias[i];
                inputs[idx] = @floatCast(f32, result);
            }
        }
    }
};

pub const CausalSelfAttention = struct {
    const Self = @This();

    n_heads: usize,
    n_embed: usize,
    head_dim: usize,
    c_attn: Linear,
    c_proj: Linear,

    pub fn init(n_heads: usize, n_embed: usize, c_attn: Linear, c_proj: Linear) Self {
        return Self{
            .n_heads = n_heads,
            .n_embed = n_embed,
            .head_dim = n_embed / n_heads,
            .c_attn = c_attn,
            .c_proj = c_proj,
        };
    }

    pub fn forward(
        self: Self,
        seq_len: usize,
        inputs: []const f32,
        outputs: []f32,
        // Parameters below are intermediate buffers used inside the function.
        _qkv: []f32,
        _q: []f32,
        _k: []f32,
        _v: []f32,
        _attn: []f32,
    ) void {
        self.c_attn.forward(inputs, _qkv);
        self.split_qkv(seq_len, _qkv, 0, outputs);
        self.transpose(seq_len, outputs, _q);
        self.split_qkv(seq_len, _qkv, 1, outputs);
        self.transpose(seq_len, outputs, _k);
        self.split_qkv(seq_len, _qkv, 2, outputs);
        self.transpose(seq_len, outputs, _v);
        scaled_dot_product_attention(
            _q,
            _k,
            _v,
            self.n_heads,
            seq_len,
            self.head_dim,
            outputs,
            _attn,
        );
        // Hack: Store untranspose requst in q so we don't need to keep another buffer.
        self.untranspose(seq_len, outputs, _q);
        self.c_proj.forward(_q, outputs);
    }

    // Splits (batch_size, seq_len, 3 * n_embed) -> (batch_size, n_heads, n_embed). The split_index
    // determines which split to return.
    pub fn split_qkv(
        self: Self,
        seq_len: usize,
        inputs: []const f32,
        split_idx: usize,
        outputs: []f32,
    ) void {
        const n_embed_ = 3 * self.n_embed;
        const batch_size = inputs.len / (seq_len * n_embed_);
        for (0..batch_size) |b| {
            for (0..seq_len) |r| {
                const out_offset = (b * seq_len * self.n_embed) + (r * self.n_embed);
                const in_offset = (b * seq_len * n_embed_) + (r * n_embed_) + (split_idx * self.n_embed);
                std.mem.copy(
                    f32,
                    outputs[out_offset .. out_offset + self.n_embed],
                    inputs[in_offset .. in_offset + self.n_embed],
                );
            }
        }
    }

    // Transposes (batch_size, seq_len, n_heads, head_dim) -> (batch_size, n_heads, seq_len, head_dim).
    pub fn transpose(self: Self, seq_len: usize, inputs: []const f32, outputs: []f32) void {
        const batch_size = inputs.len / (seq_len * self.n_embed);
        for (0..batch_size) |b| {
            for (0..self.n_heads) |h| {
                for (0..seq_len) |r| {
                    const out_offset = (b * seq_len * self.n_embed) + (h * seq_len * self.head_dim) + (r * self.head_dim);
                    const in_offset = (b * seq_len * self.n_embed) + (r * self.n_embed) + (h * self.head_dim);
                    std.mem.copy(
                        f32,
                        outputs[out_offset .. out_offset + self.head_dim],
                        inputs[in_offset .. in_offset + self.head_dim],
                    );
                }
            }
        }
    }

    // Transposes (batch_size, n_heads, seq_len, head_dim) -> (batch_size, seq_len, n_heads, head_dim).
    pub fn untranspose(self: Self, seq_len: usize, inputs: []const f32, outputs: []f32) void {
        const batch_size = inputs.len / (seq_len * self.n_embed);
        for (0..batch_size) |b| {
            for (0..seq_len) |r| {
                for (0..self.n_heads) |h| {
                    const out_offset = (b * seq_len * self.n_embed) + (r * self.n_embed) + (h * self.head_dim);
                    const in_offset = (b * seq_len * self.n_embed) + (h * seq_len * self.head_dim) + (r * self.head_dim);
                    std.mem.copy(
                        f32,
                        outputs[out_offset .. out_offset + self.head_dim],
                        inputs[in_offset .. in_offset + self.head_dim],
                    );
                }
            }
        }
    }
};

/// Computes the Gaussian Error Linear Unit (GELU) activation function on the given inputs
/// tensor inplace. Copied from the nanogpt repo and identical to OpenAI GPT2 implementation.
/// Paper: https://arxiv.org/abs/1606.08415
pub fn gelu(inputs: []f32) void {
    for (0..inputs.len) |i| {
        const x = inputs[i];
        const z: f64 = @sqrt(2.0 / std.math.pi);
        const erf: f64 = std.math.tanh(z * (x + 0.044715 * std.math.pow(f64, x, 3.0)));
        inputs[i] = @floatCast(f32, 0.5 * x * (1.0 + erf));
    }
}

/// Computes the (stable) softmax of the given inputs tensor inplace. We assume tensor has shape
/// [batch_size, D] and compute the softmax along D.
pub fn softmax(n_features: usize, inputs: []f32) void {
    const batch_size = inputs.len / n_features;

    // TODO(eugenhotaj): Vectorize these row-wise operations.
    for (0..batch_size) |b| {
        const max = std.mem.max(f32, inputs[(b * n_features) .. (b + 1) * n_features]);

        var sum: f64 = 0.0;
        for (0..n_features) |i| {
            const idx = b * n_features + i;
            inputs[idx] = @exp(inputs[idx] - max);
            sum += inputs[idx];
        }
        for (0..n_features) |i| {
            inputs[b * n_features + i] /= @floatCast(f32, sum);
        }
    }
}

// TODO(eugenhotaj): Expose whether to apply causal attention masking as an argument.
/// Computes the causal self attention of the given q, k, v tensors.
pub fn scaled_dot_product_attention(
    q: []const f32,
    k: []const f32,
    v: []const f32,
    n_heads: usize,
    seq_len: usize,
    head_dim: usize,
    outputs: []f32,
    _attn: []f32, // Intermediate buffers used inside the function.
) void {
    const batch_size = k.len / (n_heads * seq_len * head_dim);
    for (0..batch_size) |b| {
        for (0..n_heads) |h| {
            const kqvo_offset = (b * n_heads * seq_len * head_dim) + (h * seq_len * head_dim);
            const attn_offset = (b * n_heads * seq_len * seq_len) + (h * seq_len * seq_len);

            // Compute unscaled attention weights, i.e. attn = q @ k.T.
            var q_slice = q[kqvo_offset .. kqvo_offset + seq_len * head_dim];
            var k_slice = k[kqvo_offset .. kqvo_offset + seq_len * head_dim];
            var attn_slice = _attn[attn_offset .. attn_offset + seq_len * seq_len];
            blas.cblas_sgemm(
                blas.CblasRowMajor,
                blas.CblasNoTrans,
                blas.CblasTrans,
                @intCast(i32, seq_len),
                @intCast(i32, seq_len),
                @intCast(i32, head_dim),
                1.0,
                q_slice.ptr,
                @intCast(i32, head_dim),
                k_slice.ptr,
                @intCast(i32, head_dim),
                0.0,
                attn_slice.ptr,
                @intCast(i32, seq_len),
            );

            // Compute scaled attention weights, i.e. attn = softmax(attn / sqrt(head_dim)).
            // TODO(eugenhotaj): Can we vectorize this?
            for (0..seq_len) |r| {
                for (0..seq_len) |c| {
                    const idx = r * seq_len + c;
                    if (c > r) {
                        attn_slice[idx] = -std.math.inf(f32);
                    } else {
                        attn_slice[idx] /= @sqrt(@intToFloat(f32, head_dim));
                    }
                }
                softmax(seq_len, attn_slice[r * seq_len .. (r + 1) * seq_len]);
            }

            // Compute attn @ v.
            var v_slice = v[kqvo_offset .. kqvo_offset + seq_len * head_dim];
            var out_slice = outputs[kqvo_offset .. kqvo_offset + seq_len * head_dim];
            blas.cblas_sgemm(
                blas.CblasRowMajor,
                blas.CblasNoTrans,
                blas.CblasNoTrans,
                @intCast(i32, seq_len),
                @intCast(i32, head_dim),
                @intCast(i32, seq_len),
                1.0,
                attn_slice.ptr,
                @intCast(i32, seq_len),
                v_slice.ptr,
                @intCast(i32, head_dim),
                0.0,
                out_slice.ptr,
                @intCast(i32, head_dim),
            );
        }
    }
}

pub fn load_tensor(path: []const u8, shape: []const usize, comptime dtype: type, allocator: std.mem.Allocator) ![]dtype {
    var n_elements: usize = 1;
    for (shape) |item| {
        n_elements *= item;
    }
    var tensor = try allocator.alloc(dtype, n_elements);

    const fd = try std.fs.cwd().openFile(path, .{});
    defer fd.close();
    _ = try fd.readAll(std.mem.sliceAsBytes(tensor));
    return tensor;
}
