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

pub fn Block(comptime n_heads: usize, comptime seq_len: usize, comptime head_dim: usize) type {
    const n_embed = n_heads * head_dim;
    const ln_t = ops.LayerNorm(n_embed);
    const attn_t = ops.CausalSelfAttention(n_heads, seq_len, head_dim);
    const mlp_t = MLP(n_embed);

    return struct {
        const Self = @This();
        ln_1: ln_t,
        attn: attn_t,
        ln_2: ln_t,
        mlp: mlp_t,

        pub fn init(
            ln_1_weight: []const f32,
            ln_1_bias: []const f32,
            c_attn_weight: []const f32,
            c_attn_bias: []const f32,
            c_proj_weight: []const f32,
            c_proj_bias: []const f32,
            ln_2_weight: []const f32,
            ln_2_bias: []const f32,
            c_fc_weight: []const f32,
            c_fc_bias: []const f32,
            mlp_c_proj_weight: []const f32,
            mlp_c_proj_bias: []const f32,
        ) Self {
            return Self{
                .ln_1 = ln_t.init(ln_1_weight, ln_1_bias),
                .attn = attn_t.init(c_attn_weight, c_attn_bias, c_proj_weight, c_proj_bias),
                .ln_2 = ln_t.init(ln_2_weight, ln_2_bias),
                .mlp = mlp_t.init(c_fc_weight, c_fc_bias, mlp_c_proj_weight, mlp_c_proj_bias),
            };
        }

        pub fn forward(self: Self, inputs: []f32, allocator: *const std.mem.Allocator) ![]f32 {
            // Create a copy of x for residual computation.
            var x = try allocator.alloc(f32, inputs.len);
            std.mem.copyForwards(f32, x, inputs);

            self.ln_1.forward(x);
            x = try self.attn.forward(x, allocator);
            for (0..x.len) |i| {
                x[i] += inputs[i];
                inputs[i] = x[i];
            }
            self.ln_2.forward(x);
            x = try self.mlp.forward(x, allocator);
            for (0..x.len) |i| {
                x[i] += inputs[i];
            }
            return x;
        }
    };
}

pub fn expectTensorsApproxEqual(expected: []const f32, actual: []const f32) !void {
    for (0..expected.len) |i| {
        try std.testing.expectApproxEqAbs(
            expected[i],
            actual[i],
            7e-5,
        );
    }
}

pub fn main() !void {
    const n_heads = 12;
    const head_dim = 64;
    const n_embed = n_heads * head_dim;
    const batch_size = 3;
    const seq_len = 5;

    const allocator = std.heap.page_allocator;
    const ln_1_weight = try ops.load_tensor(
        "models/124M/raw/model-h0-ln_1-b",
        &[_]usize{n_embed},
        f32,
        &allocator,
    );
    defer allocator.free(ln_1_weight);
    const ln_1_bias = try ops.load_tensor(
        "models/124M/raw/model-h0-ln_1-g",
        &[_]usize{n_embed},
        f32,
        &allocator,
    );
    defer allocator.free(ln_1_bias);
    var c_attn_weight = try ops.load_tensor(
        "models/124M/raw/model-h0-attn-c_attn-w",
        &[_]usize{ n_embed, 3 * n_embed },
        f32,
        &allocator,
    );
    defer allocator.free(c_attn_weight);
    var c_attn_bias = try ops.load_tensor(
        "models/124M/raw/model-h0-attn-c_attn-b",
        &[_]usize{3 * n_embed},
        f32,
        &allocator,
    );
    var c_proj_weight = try ops.load_tensor(
        "models/124M/raw/model-h0-attn-c_proj-w",
        &[_]usize{ n_embed, n_embed },
        f32,
        &allocator,
    );
    defer allocator.free(c_proj_weight);
    var c_proj_bias = try ops.load_tensor(
        "models/124M/raw/model-h0-attn-c_proj-b",
        &[_]usize{n_embed},
        f32,
        &allocator,
    );
    defer allocator.free(c_proj_bias);
    const ln_2_weight = try ops.load_tensor(
        "models/124M/raw/model-h0-ln_2-b",
        &[_]usize{n_embed},
        f32,
        &allocator,
    );
    defer allocator.free(ln_2_weight);
    const ln_2_bias = try ops.load_tensor(
        "models/124M/raw/model-h0-ln_2-g",
        &[_]usize{n_embed},
        f32,
        &allocator,
    );
    defer allocator.free(ln_2_bias);
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
    const mlp_c_proj_weight = try ops.load_tensor(
        "models/124M/raw/model-h0-mlp-c_proj-w",
        &[_]usize{ 4 * n_embed, n_embed },
        f32,
        &allocator,
    );
    defer allocator.free(mlp_c_proj_weight);
    const mlp_c_proj_bias = try ops.load_tensor(
        "models/124M/raw/model-h0-mlp-c_proj-b",
        &[_]usize{n_embed},
        f32,
        &allocator,
    );
    defer allocator.free(mlp_c_proj_bias);
    const inputs = try ops.load_tensor(
        "models/test/gpt_inputs",
        &[_]usize{ batch_size, seq_len, n_embed },
        f32,
        &allocator,
    );
    defer allocator.free(inputs);
    const expected = try ops.load_tensor(
        "models/test/gpt_outputs",
        &[_]usize{ batch_size, seq_len, n_embed },
        f32,
        &allocator,
    );
    defer allocator.free(expected);

    const block = Block(n_heads, seq_len, head_dim).init(
        ln_1_weight,
        ln_1_bias,
        c_attn_weight,
        c_attn_bias,
        c_proj_weight,
        c_proj_bias,
        ln_2_weight,
        ln_2_bias,
        c_fc_weight,
        c_fc_bias,
        mlp_c_proj_weight,
        mlp_c_proj_bias,
    );
    const actual = try block.forward(inputs, &allocator);
    defer allocator.free(actual);

    try expectTensorsApproxEqual(expected, actual);
}
