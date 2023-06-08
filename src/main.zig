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
    const shape = [_]usize{ 786, 3072 };
    const tensor = try load_tensor(w_path, &shape, &allocator);
    std.debug.print("{any}\n", .{tensor[0..10].*});
    std.debug.print("{any}\n", .{tensor[2359286..2359296].*});
}

// pub fn main() !void {
//     var allocator = std.heap.page_allocator;

//     var weight = [_]f32{ -0.5384, 0.3873, 0.0247, -0.0153, 0.0466, 0.0597, -0.3315, -0.5699, -0.3644 };
//     var bias = [_]f32{ 0.0489, -0.0189, 0.0355 };
//     const layer = Linear(3, 3).init(&weight, &bias);

//     var inputs = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };

//     const outputs = try layer.forward(&inputs, &allocator);
//     std.debug.print("{any}\n", .{outputs.*});
// }
