const std = @import("std");
const c = @cImport(@cInclude("Accelerate/Accelerate.h"));

pub const Linear = struct {
    const Self = @This();

    in_features: usize,
    out_features: usize,
    weight: []const f32, // Weights must be provided in *column major* order!
    bias: ?[]const f32,

    pub fn init(in_features: usize, out_features: usize, weight: []const f32, bias: ?[]const f32) Self {
        return Self{
            .in_features = in_features,
            .out_features = out_features,
            .weight = weight,
            .bias = bias,
        };
    }

    pub fn forward(self: Self, inputs: []const f32, outputs: []f32) void {
        const batch_size = inputs.len / self.in_features;
        var beta: f32 = 0.0;
        if (self.bias) |bias| {
            for (0..batch_size) |b| {
                @memcpy(outputs[b * self.out_features .. (b + 1) * self.out_features], bias);
            }
            beta = 1.0;
        }
        c.cblas_sgemm(
            c.CblasRowMajor,
            c.CblasNoTrans,
            c.CblasTrans,
            @intCast(batch_size),
            @intCast(self.out_features),
            @intCast(self.in_features),
            1.0,
            inputs.ptr,
            @intCast(self.in_features),
            self.weight.ptr,
            @intCast(self.in_features),
            beta,
            outputs.ptr,
            @intCast(self.out_features),
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
            @memcpy(
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
            var mean: f32 = 0.0;
            var std_: f32 = 0.0;
            for (0..self.n_features) |i| {
                const x = inputs[b * self.n_features + i];
                mean += x;
                std_ += x * x;
            }
            const n: f32 = @floatFromInt(self.n_features);
            mean /= n;
            std_ = @sqrt((std_ / n) - (mean * mean) + self.eps);

            // Normalize.
            for (0..self.n_features) |i| {
                const idx = b * self.n_features + i;
                const x = inputs[idx];
                inputs[idx] = (x - mean) / std_ * self.weight[i] + self.bias[i];
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

    // TODO(eugenhotaj): Remove the batch_size == 1 restriction. We impose this restriction right
    // now because extending the KV cache for larger batch sizes is a bit tedious. It involves
    // expanding the sequence dimension which requires copying and moving around memory.
    pub fn forward(
        self: Self,
        seq_len: usize,
        inputs: []const f32,
        k_cache: []f32,
        v_cache: []f32,
        outputs: []f32,
        // Parameters below are intermediate buffers used inside the function.
        _qkv: []f32,
        _q: []f32,
        _k: []f32,
        _v: []f32,
        _attn: []f32,
    ) void {
        self.c_attn.forward(inputs, _qkv);

        // Q: 1 * n_embed.
        self.split_qkv(1, _qkv, 0, outputs);
        Self.transpose([3]usize{ 1, self.n_heads, self.head_dim }, outputs, _q);

        const t_shape = [3]usize{ seq_len, self.n_heads, self.head_dim };
        // Extend K: 1 * n_embed --> seq_len * n_embed.
        self.split_qkv(1, _qkv, 1, outputs);
        @memcpy(k_cache[(seq_len - 1) * self.n_embed .. seq_len * self.n_embed], outputs);
        Self.transpose(t_shape, k_cache, _k);

        // Extend V: 1 * n_embed --> seq_len * n_embed.
        self.split_qkv(1, _qkv, 2, outputs);
        @memcpy(v_cache[(seq_len - 1) * self.n_embed .. seq_len * self.n_embed], outputs);
        Self.transpose(t_shape, v_cache, _v);

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
        // Hack: Store untranspose in _q so we don't need to keep another buffer.
        Self.transpose([3]usize{ self.n_heads, 1, self.head_dim }, outputs, _q);
        self.c_proj.forward(_q, outputs);
    }

    /// Splits (seq_len, 3 * n_embed) -> (batch_size, n_heads, n_embed). The split_index
    /// determines which split to return.
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
                @memcpy(
                    outputs[out_offset .. out_offset + self.n_embed],
                    inputs[in_offset .. in_offset + self.n_embed],
                );
            }
        }
    }

    // Transposes (b, t, n, h) --> (b, n, t, h) where shape contains the sizes of (t, n, h).
    pub fn transpose(shape: [3]usize, inputs: []const f32, outputs: []f32) void {
        const seq_len = shape[0];
        const n_heads = shape[1];
        const head_dim = shape[2];
        const batch_size = inputs.len / (seq_len * n_heads * head_dim);
        for (0..batch_size) |b| {
            for (0..n_heads) |h| {
                for (0..seq_len) |s| {
                    const in_offset = (b * seq_len * n_heads * head_dim) + (s * n_heads * head_dim) + (h * head_dim);
                    const out_offset = (b * seq_len * n_heads * head_dim) + (h * seq_len * head_dim) + (s * head_dim);
                    @memcpy(
                        outputs[out_offset .. out_offset + head_dim],
                        inputs[in_offset .. in_offset + head_dim],
                    );
                }
            }
        }
    }
};

/// Computes Gaussian Error Linear Unit (GELU) activation on the given inputs tensor inplace.
/// Paper: https://arxiv.org/abs/1606.08415
pub fn gelu(inputs: []f32) void {
    for (0..inputs.len) |i| {
        const x = inputs[i];
        inputs[i] = 0.5 * x * (1.0 + std.math.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)));
        // Faster, but less accurate gelu.
        // inputs[i] = x / (1.0 + @exp(-1.702 * x));
    }
}

/// Computes the (stable) softmax of the given inputs vector inplace.
pub fn softmax(inputs: []f32) void {
    const max = std.mem.max(f32, inputs);
    var sum: f32 = 0.0;
    for (0..inputs.len) |i| {
        inputs[i] = @exp(inputs[i] - max);
        sum += inputs[i];
    }
    for (0..inputs.len) |i| {
        inputs[i] /= sum;
    }
}

/// Computes the scaled dot product attention.
///
/// The dimensions of the input tensors are expected to be:
///     q: batch_size * n_heads * 1 * head_dim
///     k: batch_size * n_heads * seq_len * head_dim
///     v: batch_size * n_heads * seq_len * head_dim
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
            const qo_offset = (b * n_heads * 1 * head_dim) + (h * 1 * head_dim);
            const kv_offset = (b * n_heads * seq_len * head_dim) + (h * seq_len * head_dim);

            // Compute attention logits, i.e. attn = softmax((q @ k.T) / sqrt(head_dim)).
            var q_slice = q[qo_offset .. qo_offset + 1 * head_dim];
            var k_slice = k[kv_offset .. kv_offset + seq_len * head_dim];
            c.cblas_sgemm(
                c.CblasRowMajor,
                c.CblasNoTrans,
                c.CblasTrans,
                1,
                @intCast(seq_len),
                @intCast(head_dim),
                1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
                q_slice.ptr,
                @intCast(head_dim),
                k_slice.ptr,
                @intCast(head_dim),
                0.0,
                _attn.ptr,
                @intCast(seq_len),
            );
            softmax(_attn);

            // Compute attn @ v.
            var v_slice = v[kv_offset .. kv_offset + seq_len * head_dim];
            var out_slice = outputs[qo_offset .. qo_offset + 1 * head_dim];
            c.cblas_sgemm(
                c.CblasRowMajor,
                c.CblasNoTrans,
                c.CblasNoTrans,
                1,
                @intCast(head_dim),
                @intCast(seq_len),
                1.0,
                _attn.ptr,
                @intCast(seq_len),
                v_slice.ptr,
                @intCast(head_dim),
                0.0,
                out_slice.ptr,
                @intCast(head_dim),
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

pub fn load_json(path: []const u8, allocator: std.mem.Allocator) !std.json.Value {
    const fd = try std.fs.cwd().openFile(path, .{});
    const buffer = try fd.readToEndAlloc(allocator, 4 * 1024 * 1024);
    return std.json.parseFromSliceLeaky(std.json.Value, allocator, buffer, .{});
}
