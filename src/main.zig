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

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const w_path = "models/124M/raw/model-h0-mlp-c_fc-w";
    const w_shape = [_]usize{ 3072, 768 };
    const w_tensor = try load_tensor(w_path, &w_shape, &allocator);

    const b_path = "models/124M/raw/model-h0-mlp-c_fc-w";
    const b_shape = [_]usize{3072};
    const b_tensor = try load_tensor(b_path, &b_shape, &allocator);

    const layer = Linear(768, 3072).init(w_tensor, b_tensor);

    var inputs = try allocator.alloc(f32, 5 * 768);
    var rng = std.rand.DefaultPrng.init(0);
    for (0..5 * 768) |i| {
        inputs[i] = rng.random().floatNorm(f32);
    }
    _ = try layer.forward(inputs, &allocator);
}
