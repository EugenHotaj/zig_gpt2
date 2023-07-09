const std = @import("std");
const c = @cImport(@cInclude("cblas.h"));

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

    /// Splits (batch_size, seq_len, 3 * n_embed) -> (batch_size, n_heads, n_embed). The split_index
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

    /// Transposes (batch_size, seq_len, n_heads, head_dim) -> (batch_size, n_heads, seq_len, head_dim).
    pub fn transpose(self: Self, seq_len: usize, inputs: []const f32, outputs: []f32) void {
        const batch_size = inputs.len / (seq_len * self.n_embed);
        for (0..batch_size) |b| {
            for (0..self.n_heads) |h| {
                for (0..seq_len) |r| {
                    const out_offset = (b * seq_len * self.n_embed) + (h * seq_len * self.head_dim) + (r * self.head_dim);
                    const in_offset = (b * seq_len * self.n_embed) + (r * self.n_embed) + (h * self.head_dim);
                    @memcpy(
                        outputs[out_offset .. out_offset + self.head_dim],
                        inputs[in_offset .. in_offset + self.head_dim],
                    );
                }
            }
        }
    }

    /// Transposes (batch_size, n_heads, seq_len, head_dim) -> (batch_size, seq_len, n_heads, head_dim).
    pub fn untranspose(self: Self, seq_len: usize, inputs: []const f32, outputs: []f32) void {
        const batch_size = inputs.len / (seq_len * self.n_embed);
        for (0..batch_size) |b| {
            for (0..seq_len) |r| {
                for (0..self.n_heads) |h| {
                    const out_offset = (b * seq_len * self.n_embed) + (r * self.n_embed) + (h * self.head_dim);
                    const in_offset = (b * seq_len * self.n_embed) + (h * seq_len * self.head_dim) + (r * self.head_dim);
                    @memcpy(
                        outputs[out_offset .. out_offset + self.head_dim],
                        inputs[in_offset .. in_offset + self.head_dim],
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

            // Compute attention logits, i.e. attn = (q @ k.T) / sqrt(head_dim).
            var q_slice = q[kqvo_offset .. kqvo_offset + seq_len * head_dim];
            var k_slice = k[kqvo_offset .. kqvo_offset + seq_len * head_dim];
            var attn_slice = _attn[attn_offset .. attn_offset + seq_len * seq_len];
            c.cblas_sgemm(
                c.CblasRowMajor,
                c.CblasNoTrans,
                c.CblasTrans,
                @intCast(seq_len),
                @intCast(seq_len),
                @intCast(head_dim),
                1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
                q_slice.ptr,
                @intCast(head_dim),
                k_slice.ptr,
                @intCast(head_dim),
                0.0,
                attn_slice.ptr,
                @intCast(seq_len),
            );

            // Causally mask and compute attention probabilities, i.e. attn = softmax(attn);
            for (0..seq_len) |r| {
                @memset(attn_slice[r * seq_len + r + 1 .. (r + 1) * seq_len], -std.math.inf(f32));
                softmax(attn_slice[r * seq_len .. (r + 1) * seq_len]);
            }

            // Compute attn @ v.
            var v_slice = v[kqvo_offset .. kqvo_offset + seq_len * head_dim];
            var out_slice = outputs[kqvo_offset .. kqvo_offset + seq_len * head_dim];
            c.cblas_sgemm(
                c.CblasRowMajor,
                c.CblasNoTrans,
                c.CblasNoTrans,
                @intCast(seq_len),
                @intCast(head_dim),
                @intCast(seq_len),
                1.0,
                attn_slice.ptr,
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
