const std = @import("std");

pub fn Linear(comptime in_features: usize, comptime out_features: usize) type {
    return struct {
        const Self = @This();
        weight: []const f32, // Weights must be provided in *column major* order!
        bias: []const f32,

        pub fn init(weight: []const f32, bias: []const f32) Self {
            return Self{ .weight = weight, .bias = bias };
        }

        pub fn forward(self: Self, inputs: []const f32, allocator: *const std.mem.Allocator) ![]f32 {
            const batch_size = inputs.len / in_features;
            var outputs = try allocator.alloc(f32, batch_size * out_features);
            for (0..batch_size) |b| {
                for (0..out_features) |o| {
                    var sum: f64 = 0.0;
                    for (0..in_features) |i| {
                        sum += inputs[b * in_features + i] * self.weight[o * in_features + i];
                    }
                    outputs[b * out_features + o] = @floatCast(f32, sum + self.bias[o]);
                }
            }
            return outputs;
        }
    };
}

pub fn Embedding(comptime embedding_dim: usize) type {
    return struct {
        const Self = @This();
        weight: []const f32,

        pub fn init(weight: []const f32) Self {
            return Self{ .weight = weight };
        }

        pub fn forward(self: Self, idxs: []const usize, allocator: *const std.mem.Allocator) ![]f32 {
            var embeddings = try allocator.alloc(f32, idxs.len * embedding_dim);
            for (0..idxs) |i| {
                const idx = idxs[i];
                // TODO(eugenhotaj): There is no reason to copy memory here. We should
                // instead return views (pointers) to the embeddings.
                std.mem.copyForwards(
                    f32,
                    embeddings[i * embedding_dim .. (i + 1) * embedding_dim],
                    self.weight[embedding_dim * idx .. embedding_dim * (idx + 1)],
                );
            }
            return embeddings;
        }
    };
}

pub fn LayerNorm(comptime n_features: usize) type {
    return struct {
        const Self = @This();
        weight: []const f32,
        bias: []const f32,
        eps: f32 = 1e-5,

        pub fn init(weight: []const f32, bias: []const f32) Self {
            return Self{ .weight = weight, .bias = bias };
        }

        pub fn forward(self: Self, inputs: []f32) void {
            const batch_size = inputs.len / n_features;
            for (0..batch_size) |b| {
                // Compute the mean and variance.
                var mean: f64 = 0.0;
                var std_: f64 = 0.0;
                for (0..n_features) |i| {
                    const x = inputs[b * n_features + i];
                    mean += x;
                    std_ += x * x;
                }
                const n = @intToFloat(f64, n_features);
                mean /= n;
                std_ = @sqrt((std_ / n) - (mean * mean) + self.eps);

                // Normalize.
                for (0..n_features) |i| {
                    const idx = b * n_features + i;
                    const x = inputs[idx];
                    const result = (x - mean) / std_ * self.weight[i] + self.bias[i];
                    inputs[idx] = @floatCast(f32, result);
                }
            }
        }
    };
}

/// Computes the causal self attention of the given q, k, v tensors.
pub fn CausalSelfAttention(comptime n_heads: usize, comptime seq_len: usize, comptime head_dim: usize) type {
    return struct {
        const Self = @This();
        const n_embed = n_heads * head_dim;

        c_attn: Linear(n_embed, 3 * n_embed),
        c_proj: Linear(n_embed, n_embed),

        pub fn init(
            c_attn_weight: []const f32,
            c_attn_bias: []const f32,
            c_proj_weight: []const f32,
            c_proj_bias: []const f32,
        ) Self {
            const c_attn = Linear(n_embed, 3 * n_embed).init(c_attn_weight, c_attn_bias);
            const c_proj = Linear(n_embed, n_embed).init(c_proj_weight, c_proj_bias);
            return Self{ .c_attn = c_attn, .c_proj = c_proj };
        }

        pub fn forward(self: Self, inputs: []const f32, allocator: *const std.mem.Allocator) ![]f32 {
            const qkv = try self.c_attn.forward(inputs, allocator);
            const q = try Self.transpose(
                try Self.split_qkv(qkv, 0, allocator),
                allocator,
            );
            const k = try Self.transpose(
                try Self.split_qkv(qkv, 1, allocator),
                allocator,
            );
            const v = try Self.transpose(
                try Self.split_qkv(qkv, 2, allocator),
                allocator,
            );
            const outputs = try Self.untranspose(
                try scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    n_heads,
                    seq_len,
                    head_dim,
                    allocator,
                ),
                allocator,
            );
            return self.c_proj.forward(outputs, allocator);
        }

        // Splits (batch_size, seq_len, 3 * n_embed) -> (batch_size, n_heads, n_embed). The split_index
        // determines which split to return.
        pub fn split_qkv(inputs: []const f32, split_idx: usize, allocator: *const std.mem.Allocator) ![]f32 {
            const n_embed_ = 3 * n_embed;
            const batch_size = inputs.len / (seq_len * n_embed_);
            var outputs = try allocator.alloc(f32, inputs.len / 3);
            for (0..batch_size) |b| {
                for (0..seq_len) |r| {
                    const out_offset = (b * seq_len * n_embed) + (r * n_embed);
                    const in_offset = (b * seq_len * n_embed_) + (r * n_embed_) + (split_idx * n_embed);
                    std.mem.copy(
                        f32,
                        outputs[out_offset .. out_offset + n_embed],
                        inputs[in_offset .. in_offset + n_embed],
                    );
                }
            }
            return outputs;
        }

        // Transposes (batch_size, seq_len, n_heads, head_dim) -> (batch_size, n_heads, seq_len, head_dim).
        pub fn transpose(inputs: []const f32, allocator: *const std.mem.Allocator) ![]f32 {
            const batch_size = inputs.len / (seq_len * n_embed);
            var outputs = try allocator.alloc(f32, inputs.len);
            for (0..batch_size) |b| {
                for (0..n_heads) |h| {
                    for (0..seq_len) |r| {
                        const out_offset = (b * seq_len * n_embed) + (h * seq_len * head_dim) + (r * head_dim);
                        const in_offset = (b * seq_len * n_embed) + (r * n_embed) + (h * head_dim);
                        std.mem.copy(
                            f32,
                            outputs[out_offset .. out_offset + head_dim],
                            inputs[in_offset .. in_offset + head_dim],
                        );
                    }
                }
            }
            return outputs;
        }

        // Transposes (batch_size, n_heads, seq_len, head_dim) -> (batch_size, seq_len, n_heads, head_dim).
        pub fn untranspose(inputs: []const f32, allocator: *const std.mem.Allocator) ![]f32 {
            const batch_size = inputs.len / (seq_len * n_embed);
            var outputs = try allocator.alloc(f32, inputs.len);
            for (0..batch_size) |b| {
                for (0..seq_len) |r| {
                    for (0..n_heads) |h| {
                        const out_offset = (b * seq_len * n_embed) + (r * n_embed) + (h * head_dim);
                        const in_offset = (b * seq_len * n_embed) + (h * seq_len * head_dim) + (r * head_dim);
                        std.mem.copy(
                            f32,
                            outputs[out_offset .. out_offset + head_dim],
                            inputs[in_offset .. in_offset + head_dim],
                        );
                    }
                }
            }
            return outputs;
        }
    };
}

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
    allocator: *const std.mem.Allocator,
) ![]f32 {
    const batch_size = k.len / (n_heads * seq_len * head_dim);

    var attn = try allocator.alloc(f32, batch_size * n_heads * seq_len * seq_len);
    defer allocator.free(attn);
    var outputs = try allocator.alloc(f32, batch_size * n_heads * seq_len * head_dim);
    for (0..batch_size) |b| {
        for (0..n_heads) |h| {
            const in_offset = (b * n_heads * seq_len * head_dim) + (h * seq_len * head_dim);
            const out_offset = (b * n_heads * seq_len * seq_len) + (h * seq_len * seq_len);

            for (0..seq_len) |r| {
                // Compute attention weights, i.e. attn = softmax(q @ k.T / head_dim).
                for (0..seq_len) |c| {
                    // For masked elements, short-circut the matmul and directly apply
                    // the mask.
                    if (c > r) {
                        attn[out_offset + r * seq_len + c] = -std.math.inf(f32);
                        continue;
                    }

                    // Otherwise compute (q @ k.T) / sqrt(head_dim).
                    var sum: f64 = 0.0;
                    for (0..head_dim) |i| {
                        sum += q[in_offset + r * head_dim + i] * k[in_offset + c * head_dim + i];
                    }
                    const value = sum / @sqrt(@intToFloat(f64, head_dim));
                    attn[out_offset + r * seq_len + c] = @floatCast(f32, value);
                }
                const offset = out_offset + r * seq_len;
                softmax(seq_len, attn[offset .. offset + seq_len]);

                // Compute attn @ v.
                for (0..head_dim) |c| {
                    var sum: f64 = 0.0;
                    for (0..seq_len) |i| {
                        // TODO(eugenhotaj): Not cache friendly.
                        sum += attn[out_offset + r * seq_len + i] * v[in_offset + i * head_dim + c];
                    }
                    outputs[in_offset + r * head_dim + c] = @floatCast(f32, sum);
                }
            }
        }
    }
    return outputs;
}

pub fn load_tensor(path: []const u8, shape: []const usize, comptime dtype: type, allocator: *const std.mem.Allocator) ![]dtype {
    var n_elements: usize = @alignOf(dtype);
    for (shape) |item| {
        n_elements *= item;
    }

    const fd = try std.fs.cwd().openFile(path, .{});
    defer fd.close();

    var tensor = try allocator.alloc(u8, n_elements);
    _ = try fd.readAll(tensor);
    return std.mem.bytesAsSlice(dtype, @alignCast(@alignOf(dtype), tensor));
}
