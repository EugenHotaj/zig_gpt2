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

/// Computes the (stable) softmax of the given tensor inplace. We assume tensor has shape
/// [batch_size, D] and compute the softmax along D.
pub fn softmax(batch_size: usize, tensor: *[]f32) void {
    const n_features = tensor.len / batch_size;

    // TODO(eugenhotaj): Vectorize these row-wise operations.
    for (0..batch_size) |b| {
        const max = std.mem.max(f32, tensor.*[(b * n_features) .. (b + 1) * n_features]);

        var sum: f32 = 0.0;
        for (0..n_features) |i| {
            const idx = b * n_features + i;
            tensor.*[idx] = std.math.exp(tensor.*[idx] - max);
            sum += tensor.*[idx];
        }
        for (0..n_features) |i| {
            tensor.*[b * n_features + i] /= sum;
        }
    }
}

/// Computes the Gaussian Error Linear Unit (GELU) activation function on the given tensor inplace.
/// Copied from the nanogpt repo and identical to OpenAI GPT2 implementation.
/// Paper: https://arxiv.org/abs/1606.08415
pub fn gelu(inputs: *[]f32) void {
    for (0..inputs.len) |i| {
        const x = inputs.*[i];
        const z = std.math.sqrt(2.0 / std.math.pi);
        const erf = std.math.tanh(z * (x + 0.044715 * std.math.pow(f32, x, 3.0)));
        inputs.*[i] = 0.5 * x * (1.0 + erf);
    }
}

pub fn load_tensor(path: []const u8, shape: []const usize, comptime dtype: type, allocator: *const std.mem.Allocator) ![]dtype {
    var n_elements: usize = @alignOf(dtype); // Using 4 since we're loading f32s (4 bytes).
    for (shape) |item| {
        n_elements *= item;
    }

    const fd = try std.fs.cwd().openFile(path, .{});
    defer fd.close();

    var tensor = try allocator.alloc(u8, n_elements);
    _ = try fd.readAll(tensor);
    return std.mem.bytesAsSlice(dtype, @alignCast(@alignOf(dtype), tensor));
}
