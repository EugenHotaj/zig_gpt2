const std = @import("std");
const ops = @import("ops.zig");

const GPTConfig = struct {
    const Self = @This();

    n_heads: usize,
    seq_len: usize,
    head_dim: usize,
    n_embed: usize,

    pub fn init(n_heads: usize, seq_len: usize, head_dim: usize) Self {
        const n_embed = n_heads * head_dim;
        return Self{ .n_heads = n_heads, .seq_len = seq_len, .head_dim = head_dim, .n_embed = n_embed };
    }
};

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

pub fn load_linear(
    name: []const u8,
    in_features: usize,
    out_features: usize,
    allocator: std.mem.Allocator,
) !ops.Linear {
    const weight_path = try std.fmt.allocPrint(allocator, "models/124M/raw/model-{s}-w", .{name});
    defer allocator.free(weight_path);
    var weight = try ops.load_tensor(
        weight_path,
        &[_]usize{ in_features, out_features },
        f32,
        &allocator,
    );
    const bias_path = try std.fmt.allocPrint(allocator, "models/124M/raw/model-{s}-b", .{name});
    defer allocator.free(bias_path);
    var bias = try ops.load_tensor(
        bias_path,
        &[_]usize{out_features},
        f32,
        &allocator,
    );
    return ops.Linear.init(in_features, out_features, weight, bias);
}

pub fn load_layer_norm(
    name: []const u8,
    n_features: usize,
    allocator: std.mem.Allocator,
) !ops.LayerNorm {
    const weight_path = try std.fmt.allocPrint(allocator, "models/124M/raw/model-{s}-b", .{name});
    defer allocator.free(weight_path);
    var weight = try ops.load_tensor(
        weight_path,
        &[_]usize{n_features},
        f32,
        &allocator,
    );
    const bias_path = try std.fmt.allocPrint(allocator, "models/124M/raw/model-{s}-g", .{name});
    defer allocator.free(bias_path);
    var bias = try ops.load_tensor(
        bias_path,
        &[_]usize{n_features},
        f32,
        &allocator,
    );
    return ops.LayerNorm.init(n_features, weight, bias);
}

pub fn load_block(layer_idx: usize, config: GPTConfig, allocator: std.mem.Allocator) !Block {
    const ln_1_name = try std.fmt.allocPrint(allocator, "h{any}-ln_1", .{layer_idx});
    defer allocator.free(ln_1_name);
    const ln_1 = try load_layer_norm(ln_1_name, config.n_embed, allocator);

    const c_attn_name = try std.fmt.allocPrint(allocator, "h{any}-attn-c_attn", .{layer_idx});
    defer allocator.free(c_attn_name);
    const c_attn = try load_linear(c_attn_name, config.n_embed, 3 * config.n_embed, allocator);

    const c_proj_name = try std.fmt.allocPrint(allocator, "h{any}-attn-c_proj", .{layer_idx});
    defer allocator.free(c_proj_name);
    const c_proj = try load_linear(c_proj_name, config.n_embed, config.n_embed, allocator);

    const ln_2_name = try std.fmt.allocPrint(allocator, "h{any}-ln_2", .{layer_idx});
    defer allocator.free(ln_2_name);
    const ln_2 = try load_layer_norm(ln_2_name, config.n_embed, allocator);

    const c_fc_name = try std.fmt.allocPrint(allocator, "h{any}-mlp-c_fc", .{layer_idx});
    defer allocator.free(c_fc_name);
    const c_fc = try load_linear(c_fc_name, config.n_embed, 4 * config.n_embed, allocator);

    const mlp_c_proj_name = try std.fmt.allocPrint(allocator, "h{any}-mlp-c_proj", .{layer_idx});
    defer allocator.free(mlp_c_proj_name);
    const mlp_c_proj = try load_linear(mlp_c_proj_name, 4 * config.n_embed, config.n_embed, allocator);

    const attn = ops.CausalSelfAttention.init(config.n_heads, config.seq_len, config.head_dim, c_attn, c_proj);
    const mlp = MLP.init(c_fc, mlp_c_proj);
    return Block.init(ln_1, attn, ln_2, mlp);
}

pub fn main() !void {
    const batch_size = 3;
    const config = GPTConfig.init(12, 5, 64);

    const allocator = std.heap.page_allocator;
    const block = try load_block(0, config, allocator);
    const inputs = try ops.load_tensor(
        "models/test/gpt_inputs",
        &[_]usize{ batch_size, config.seq_len, config.n_embed },
        f32,
        &allocator,
    );
    defer allocator.free(inputs);
    const expected = try ops.load_tensor(
        "models/test/gpt_outputs",
        &[_]usize{ batch_size, config.seq_len, config.n_embed },
        f32,
        &allocator,
    );
    defer allocator.free(expected);

    const actual = try block.forward(inputs, &allocator);
    defer allocator.free(actual);

    try expectTensorsApproxEqual(expected, actual);
}
