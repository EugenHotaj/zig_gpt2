const std = @import("std");
const ops = @import("ops.zig");
const expectTensorsApproxEqual = @import("tests.zig").expectTensorsApproxEqual;

const GPTConfig = struct {
    const Self = @This();

    vocab_size: usize,
    context_size: usize,
    n_layer: usize,
    n_heads: usize,
    n_embed: usize,

    pub fn init(vocab_size: usize, context_size: usize, n_layer: usize, n_heads: usize, n_embed: usize) Self {
        return Self{
            .vocab_size = vocab_size,
            .context_size = context_size,
            .n_layer = n_layer,
            .n_heads = n_heads,
            .n_embed = n_embed,
        };
    }
};

pub const State = struct {
    const Self = @This();

    pos: []usize,
    pos_emb: []f32,
    x: []f32,
    o: []f32,
    logits: []f32,

    // Intermediate buffers.
    _h: []f32,
    _3xh: []f32,
    _4xh: []f32,
    _attn: []f32,

    allocator: std.mem.Allocator,
    pool: ?*std.Thread.Pool,

    pub fn init(batch_size: usize, seq_len: usize, config: GPTConfig, allocator: std.mem.Allocator, pool: ?*std.Thread.Pool) !Self {
        var pos = try allocator.alloc(usize, seq_len);
        for (0..seq_len) |i| {
            pos[i] = i;
        }
        return Self{
            .pos = pos,
            .pos_emb = try allocator.alloc(f32, seq_len * config.n_embed),
            .x = try allocator.alloc(f32, batch_size * seq_len * config.n_embed),
            .o = try allocator.alloc(f32, batch_size * seq_len * config.n_embed),
            .logits = try allocator.alloc(f32, batch_size * config.vocab_size),
            ._h = try allocator.alloc(f32, batch_size * seq_len * config.n_embed),
            ._3xh = try allocator.alloc(f32, batch_size * seq_len * 3 * config.n_embed),
            ._4xh = try allocator.alloc(f32, batch_size * seq_len * 4 * config.n_embed),
            ._attn = try allocator.alloc(f32, batch_size * config.n_heads * seq_len * seq_len),
            .allocator = allocator,
            .pool = pool,
        };
    }
};

const MLP = struct {
    const Self = @This();

    c_fc: ops.Linear,
    c_proj: ops.Linear,

    pub fn init(c_fc: ops.Linear, c_proj: ops.Linear) MLP {
        return MLP{ .c_fc = c_fc, .c_proj = c_proj };
    }

    /// Computes the forward pass and writes the result to state.o.
    pub fn forward(self: Self, inputs: []const f32, state: State) !void {
        try self.c_fc.forward(inputs, state._4xh, state.pool);
        ops.gelu(state._4xh);
        try self.c_proj.forward(state._4xh, state.o, state.pool);
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

    /// Computes the forward pass and writes the result to both state.x and state.o. This
    /// enables sequentially calling multiple Block.forwards() in a row without having to copy
    /// memory.
    pub fn forward(self: Self, seq_len: usize, inputs: []const f32, state: State) !void {
        // Create a copy of x for residual computation.
        std.mem.copyForwards(f32, state._h, inputs);

        self.ln_1.forward(state._h);
        try self.attn.forward(
            seq_len,
            state._h,
            state.o,
            state.pool,
            state._3xh,
            // Using _4xh as temporary buffer since _q, _k, _v are thrown away.
            state._4xh[0..inputs.len],
            state._4xh[inputs.len .. 2 * inputs.len],
            state._4xh[2 * inputs.len .. 3 * inputs.len],
            state._attn,
        );
        for (0..state.o) |i| {
            state._h[i] = state.o[i] + inputs[i];
            state.x[i] = state._h[i];
        }
        self.ln_2.forward(state._h);
        try self.mlp.forward(state._h, state);
        for (0..state.o) |i| {
            state.o[i] += state.x[i];
            state.x[i] = state.o[i];
        }
    }
};

const GPT = struct {
    const Self = @This();

    config: GPTConfig,
    wte: ops.Embedding,
    wpe: ops.Embedding,
    h: []const Block,
    ln_f: ops.LayerNorm,
    lm_head: ops.Linear,

    pub fn init(
        config: GPTConfig,
        wte: ops.Embedding,
        wpe: ops.Embedding,
        h: []const Block,
        ln_f: ops.LayerNorm,
        lm_head: ops.Linear,
    ) Self {
        return Self{
            .config = config,
            .wte = wte,
            .wpe = wpe,
            .h = h,
            .ln_f = ln_f,
            .lm_head = lm_head,
        };
    }

    /// Computes the forward pass and writes the result in state.logits.
    pub fn forward(self: Self, seq_len: usize, inputs: []const usize, state: State) !void {
        const batch_size = inputs.len / seq_len;

        // Compute embeddings (token + positional).
        self.wpe.forward(state.pos, state.pos_emb);
        self.wte.forward(inputs, state.x);
        for (0..batch_size) |b| {
            for (0..seq_len) |s| {
                for (0..self.config.n_embed) |i| {
                    const batch_offset = (b * seq_len * self.config.n_embed);
                    const seq_offset = (s * self.config.n_embed);
                    state.x[batch_offset + seq_offset + i] += state.pos_emb[seq_offset + i];
                }
            }
        }

        // Forward the transformer.
        for (0..self.h.len) |i| {
            try self.h[i].forward(seq_len, state.x, state);
        }
        self.ln_f.forward(state.x);

        // Compute logits.
        // Mini-optimization: Only forward the lm_head on the very last position.
        for (0..batch_size) |b| {
            const in_offset = b * seq_len * self.config.n_embed;
            try self.lm_head.forward(
                state.x[in_offset + (seq_len - 1) * self.config.n_embed .. in_offset + seq_len * self.config.n_embed],
                state.logits[b * self.config.vocab_size .. (b + 1) * self.config.vocab_size],
                state.pool,
            );
        }
    }

    pub fn generate(
        self: Self,
        input_tokens: usize,
        max_tokens: usize,
        temp: f32,
        inputs: []usize,
        state: State,
    ) !void {
        const total_tokens = input_tokens + max_tokens;
        if (total_tokens > self.config.context_size) {
            return error.SequenceTooLong;
        }
        // TODO(eugenhotaj): Remove batch size = 1 restrictions.
        if ((inputs.len / total_tokens) > 1) {
            return error.BatchSizeTooBig;
        }

        const logits_dim = self.config.vocab_size;
        for (input_tokens..total_tokens) |s| {
            try self.forward(s, inputs[0..s], state);
            for (0..state.logits.len) |i| {
                state.logits[i] /= temp;
            }
            ops.softmax(logits_dim, state.logits);

            var rng = std.rand.DefaultPrng.init(@intCast(u64, std.time.timestamp()));
            var random = rng.random();
            inputs[s] = random.weightedIndex(f32, state.logits);
        }
    }
};

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
        allocator,
    );
    const bias_path = try std.fmt.allocPrint(allocator, "models/124M/raw/model-{s}-b", .{name});
    defer allocator.free(bias_path);
    var bias = try ops.load_tensor(
        bias_path,
        &[_]usize{out_features},
        f32,
        allocator,
    );
    return ops.Linear.init(in_features, out_features, weight, bias);
}

pub fn load_layer_norm(
    name: []const u8,
    n_features: usize,
    allocator: std.mem.Allocator,
) !ops.LayerNorm {
    const weight_path = try std.fmt.allocPrint(allocator, "models/124M/raw/model-{s}-g", .{name});
    defer allocator.free(weight_path);
    var weight = try ops.load_tensor(
        weight_path,
        &[_]usize{n_features},
        f32,
        allocator,
    );
    const bias_path = try std.fmt.allocPrint(allocator, "models/124M/raw/model-{s}-b", .{name});
    defer allocator.free(bias_path);
    var bias = try ops.load_tensor(
        bias_path,
        &[_]usize{n_features},
        f32,
        allocator,
    );
    return ops.LayerNorm.init(n_features, weight, bias);
}

pub fn load_embedding(name: []const u8, vocab_size: usize, emb_dim: usize, allocator: std.mem.Allocator) !ops.Embedding {
    const path = try std.fmt.allocPrint(allocator, "models/124M/raw/model-{s}", .{name});
    defer allocator.free(path);
    var weight = try ops.load_tensor(
        path,
        &[_]usize{ vocab_size, emb_dim },
        f32,
        allocator,
    );
    return ops.Embedding.init(emb_dim, weight);
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

    const attn = ops.CausalSelfAttention.init(config.n_heads, config.n_embed, c_attn, c_proj);
    const mlp = MLP.init(c_fc, mlp_c_proj);
    return Block.init(ln_1, attn, ln_2, mlp);
}

pub fn load_gpt(config: GPTConfig, allocator: std.mem.Allocator) !GPT {
    var wte = try load_embedding("wte", config.vocab_size, config.n_embed, allocator);
    const wpe = try load_embedding("wpe", config.context_size, config.n_embed, allocator);
    var h = try allocator.alloc(Block, config.n_layer);
    for (0..h.len) |i| {
        h[i] = try load_block(i, config, allocator);
    }
    const ln_f = try load_layer_norm("ln_f", config.n_embed, allocator);
    const lm_head = ops.Linear.init_no_bias(config.n_embed, config.vocab_size, wte.weight);
    return GPT.init(config, wte, wpe, h, ln_f, lm_head);
}

pub fn main() !void {
    const batch_size = 1;
    const input_tokens = 8;
    const max_tokens = 40;
    const temp = 0.8;

    const config = GPTConfig.init(50257, 1024, 12, 12, 768);

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();
    const allocator = arena.allocator();
    const inputs = try ops.load_tensor(
        "models/test/gpt_inputs",
        &[_]usize{ batch_size, input_tokens + max_tokens },
        usize,
        allocator,
    );
    const expected = try ops.load_tensor(
        "models/test/gpt_outputs",
        &[_]usize{ batch_size, 1, config.vocab_size },
        f32,
        allocator,
    );
    var pool: std.Thread.Pool = undefined;
    try std.Thread.Pool.init(&pool, .{ .allocator = allocator });
    var state = try State.init(batch_size, input_tokens + max_tokens, config, allocator, &pool);

    // Ensure that forwarding the model produces the same outputs as PyTorch.
    const gpt = try load_gpt(config, allocator);
    try gpt.forward(input_tokens, inputs[0 .. batch_size * input_tokens], state);
    try expectTensorsApproxEqual(expected, state.logits);

    // Generate.
    try gpt.generate(input_tokens, max_tokens, temp, inputs, state);
    std.debug.print("{any}\n", .{inputs});
}
