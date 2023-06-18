const std = @import("std");
const ops = @import("ops.zig");

pub fn MLP(comptime n_embed: usize) type {
    const c_fc_t = ops.Linear(n_embed, 4 * n_embed);
    const c_proj_t = ops.Linear(4 * n_embed, n_embed);

    return struct {
        const Self = @This();
        c_fc: c_fc_t,
        c_proj: c_proj_t,

        pub fn init(
            c_fc_weight: []const f32,
            c_fc_bias: []const f32,
            c_proj_weight: []const f32,
            c_proj_bias: []const f32,
        ) Self {
            return Self{
                .c_fc = c_fc_t.init(c_fc_weight, c_fc_bias),
                .c_proj = c_proj_t.init(c_proj_weight, c_proj_bias),
            };
        }

        pub fn forward(self: Self, inputs: []const f32, allocator: *const std.mem.Allocator) ![]f32 {
            var x: []f32 = undefined;
            x = try self.c_fc.forward(inputs, allocator);
            ops.gelu(x);
            x = try self.c_proj.forward(x, allocator);
            return x;
        }
    };
}

pub fn expectTensorsApproxEqual(expected: []const f32, actual: []const f32) !void {
    for (0..expected.len) |i| {
        try std.testing.expectApproxEqAbs(
            expected[i],
            actual[i],
            5e-6,
        );
    }
}

pub fn main() !void {
    const n_embed = 768;

    const allocator = std.heap.page_allocator;
    const c_fc_weight = try ops.load_tensor(
        "models/124M/raw/model-h0-mlp-c_fc-w",
        &[_]usize{ n_embed, 4 * n_embed },
        f32,
        &allocator,
    );
    defer allocator.free(c_fc_weight);
    const c_fc_bias = try ops.load_tensor(
        "models/124M/raw/model-h0-mlp-c_fc-b",
        &[_]usize{4 * n_embed},
        f32,
        &allocator,
    );
    defer allocator.free(c_fc_bias);
    const c_proj_weight = try ops.load_tensor(
        "models/124M/raw/model-h0-mlp-c_proj-w",
        &[_]usize{ 4 * n_embed, n_embed },
        f32,
        &allocator,
    );
    defer allocator.free(c_proj_weight);
    const c_proj_bias = try ops.load_tensor(
        "models/124M/raw/model-h0-mlp-c_proj-b",
        &[_]usize{n_embed},
        f32,
        &allocator,
    );
    defer allocator.free(c_proj_bias);
    const inputs = try ops.load_tensor(
        "models/test/gpt_inputs",
        &[_]usize{ 3, n_embed },
        f32,
        &allocator,
    );
    defer allocator.free(inputs);
    const expected = try ops.load_tensor(
        "models/test/gpt_outputs",
        &[_]usize{ 3, n_embed },
        f32,
        &allocator,
    );
    defer allocator.free(expected);

    const mlp = MLP(n_embed).init(c_fc_weight, c_fc_bias, c_proj_weight, c_proj_bias);
    const actual = try mlp.forward(inputs, &allocator);
    defer allocator.free(actual);

    try expectTensorsApproxEqual(expected, actual);
}
