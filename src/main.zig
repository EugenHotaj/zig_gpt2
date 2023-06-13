const std = @import("std");

pub fn Linear(comptime in_features: usize, comptime out_features: usize) type {
    return struct {
        const Self = @This();
        weight: []const f32, // Weights must be provided in *column major* order!
        bias: []const f32,

        pub fn init(weight: []const f32, bias: []const f32) Self {
            return Self{ .weight = weight, .bias = bias };
        }

        pub fn forward(self: Self, inputs: []f32, allocator: *const std.mem.Allocator) ![]f32 {
            const batch_size = inputs.len / in_features;
            var outputs = try allocator.alloc(f32, batch_size * out_features);
            for (0..batch_size) |b| {
                for (0..out_features) |o| {
                    var sum: f32 = 0.0;
                    for (0..in_features) |i| {
                        sum += inputs[b * in_features + i] * self.weight[o * in_features + i];
                    }
                    outputs[b * out_features + o] = sum + self.bias[o];
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
                var mean: f32 = 0.0;
                var std_: f32 = 0.0;
                for (0..n_features) |i| {
                    const x = inputs[b * n_features + i];
                    mean += x;
                    std_ += x * x;
                }
                mean /= @intToFloat(f32, n_features);
                std_ = std.math.sqrt((std_ / @intToFloat(f32, n_features)) - (mean * mean) + self.eps);

                // Normalize.
                for (0..n_features) |i| {
                    const idx = b * n_features + i;
                    const x = inputs[idx];
                    inputs[idx] = (x - mean) / std_ * self.weight[i] + self.bias[i];
                }
            }
        }
    };
}

/// Computes the Gaussian Error Linear Unit (GELU) activation function on the given inputs
/// tensor inplace. Copied from the nanogpt repo and identical to OpenAI GPT2 implementation.
/// Paper: https://arxiv.org/abs/1606.08415
pub fn gelu(inputs: []f32) void {
    for (0..inputs.len) |i| {
        const x = inputs[i];
        const z = std.math.sqrt(2.0 / std.math.pi);
        const erf = std.math.tanh(z * (x + 0.044715 * std.math.pow(f32, x, 3.0)));
        inputs[i] = 0.5 * x * (1.0 + erf);
    }
}

/// Computes the (stable) softmax of the given inputs tensor inplace. We assume tensor has shape
/// [batch_size, D] and compute the softmax along D.
pub fn softmax(n_features: usize, inputs: []f32) void {
    const batch_size = inputs.len / n_features;

    // TODO(eugenhotaj): Vectorize these row-wise operations.
    for (0..batch_size) |b| {
        const max = std.mem.max(f32, inputs[(b * n_features) .. (b + 1) * n_features]);

        var sum: f32 = 0.0;
        for (0..n_features) |i| {
            const idx = b * n_features + i;
            inputs[idx] = std.math.exp(inputs[idx] - max);
            sum += inputs[idx];
        }
        for (0..n_features) |i| {
            inputs[b * n_features + i] /= sum;
        }
    }
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
