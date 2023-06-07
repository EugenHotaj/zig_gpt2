const std = @import("std");

pub fn Linear(comptime I: usize, comptime O: usize) type {
    return struct {
        const Self = @This();
        in_features: usize = I,
        out_features: usize = O,
        weight: [I * O]f32, // Weights must be provided in *column major* order!
        bias: [O]f32,

        pub fn init(weight: *[I * O]f32, bias: *[O]f32) Self {
            // TODO(eugenhotaj): We're unnecessarily copying memory here.
            return Self{ .weight = weight.*, .bias = bias.* };
        }

        pub fn forward(self: Self, inputs: []f32, allocator: *std.mem.Allocator) !*[]f32 {
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
            return &outputs;
        }
    };
}

pub fn main() !void {
    var allocator = std.heap.page_allocator;

    var weight = [_]f32{ -0.5384, 0.3873, 0.0247, -0.0153, 0.0466, 0.0597, -0.3315, -0.5699, -0.3644 };
    var bias = [_]f32{ 0.0489, -0.0189, 0.0355 };
    const layer = Linear(3, 3).init(&weight, &bias);

    var inputs = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };

    const outputs = try layer.forward(&inputs, &allocator);
    std.debug.print("{any}\n", .{outputs.*});
}
