const std = @import("std");
const ops = @import("ops.zig");

const MLP = struct {
    const Self = @This();

    c_fc: ops.Linear,
    c_proj: ops.Linear,

    pub fn init(c_fc: ops.Linear, c_proj: ops.Linear) MLP {
        return MLP{ .c_fc = c_fc, .c_proj = c_proj };
    }

    pub fn forward(self: Self, inputs: []const f32, allocator: *const std.mem.Allocator) ![]f32 {
        var x: []f32 = try self.c_fc.forward(inputs, allocator);
        ops.gelu(x);
        return self.c_proj.forward(x, allocator);
    }
};

const Block = struct {
    const Self = @This();

    ln_1: ops.LayerNorm,
    attn: ops.CausalSelfAttention,
    ln_2: ops.LayerNorm,
    mlp: MLP,

    pub fn init(ln_1: ops.LayerNorm, attn: ops.CausalSelfAttention, ln_2: ops.LayerNorm, mlp: MLP) Self {
        return Self{ .ln_1 = ln_1, .attn = attn, .ln_2 = ln_2, .mlp = mlp };
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

    const ln_1 = ops.LayerNorm.init(n_embed, ln_1_weight, ln_1_bias);
    const ln_2 = ops.LayerNorm.init(n_embed, ln_2_weight, ln_2_bias);
    const attn = ops.CausalSelfAttention.init(
        n_heads,
        seq_len,
        head_dim,
        ops.Linear.init(n_embed, 3 * n_embed, c_attn_weight, c_attn_bias),
        ops.Linear.init(n_embed, n_embed, c_proj_weight, c_proj_bias),
    );
    const mlp = MLP.init(
        ops.Linear.init(n_embed, 4 * n_embed, c_fc_weight, c_fc_bias),
        ops.Linear.init(4 * n_embed, n_embed, mlp_c_proj_weight, mlp_c_proj_bias),
    );
    const block = Block.init(ln_1, attn, ln_2, mlp);
    const actual = try block.forward(inputs, &allocator);
    defer allocator.free(actual);

    try expectTensorsApproxEqual(expected, actual);
}
