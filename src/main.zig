const std = @import("std");

pub fn Linear(comptime I: usize, comptime O: usize) type {
    return struct {
        const Self = @This();
        in_features: usize = I,
        out_features: usize = O,
        weight: []const f32, // Weights must be provided in *column major* order!
        bias: []const f32,

        pub fn init(weight: []const f32, bias: []const f32) Self {
            return Self{ .weight = weight, .bias = bias };
        }

        pub fn forward(self: Self, inputs: []f32, allocator: *const std.mem.Allocator) ![]f32 {
            const batch_size = inputs.len / I;
            var outputs = try allocator.alloc(f32, batch_size * self.out_features);
            for (0..batch_size) |b| {
                for (0..O) |o| {
                    var sum: f32 = 0.0;
                    for (0..I) |i| {
                        sum += inputs[b * I + i] * self.weight[o * I + i];
                    }
                    outputs[b * O + o] = sum + self.bias[o];
                }
            }
            return outputs;
        }
    };
}

/// Computes the softmax of the given tensor inplace. We assume tensor has shape
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

pub fn gelu(inputs: *[]f32) void {
    for (0..inputs.len) |i| {
        const x = inputs.*[i];
        const z = std.math.sqrt(2.0 / std.math.pi);
        const erf = std.math.tanh(z * (x + 0.044715 * std.math.pow(f32, x, 3.0)));
        inputs.*[i] = 0.5 * x * (1.0 + erf);
    }
}

pub fn load_tensor(path: []const u8, shape: []const usize, allocator: *const std.mem.Allocator) ![]f32 {
    var n_elements: usize = 4; // Using 4 since we're loading f32s (4 bytes).
    for (shape) |item| {
        n_elements *= item;
    }

    const fd = try std.fs.cwd().openFile(path, .{});
    defer fd.close();

    var tensor = try allocator.alloc(u8, n_elements);
    _ = try fd.readAll(tensor);
    return std.mem.bytesAsSlice(f32, @alignCast(@alignOf(f32), tensor));
}
