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

pub fn gelu(inputs: []f32, allocator: *const std.mem.Allocator) ![]f32 {
    var outputs = try allocator.alloc(f32, inputs.len);
    for (0..inputs.len) |i| {
        const x = inputs[i];
        const z = std.math.sqrt(2.0 / std.math.pi);
        const erf = std.math.tanh(z * (x + 0.044715 * std.math.pow(f32, x, 3.0)));
        outputs[i] = 0.5 * x * (1.0 + erf);
    }
    return outputs;
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
