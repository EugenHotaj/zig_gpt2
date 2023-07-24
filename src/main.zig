const std = @import("std");
const ops = @import("ops.zig");
const bpe = @import("bpe.zig");

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

/// Structure which maintains state which is shared across all GPT layers.
pub const State = struct {
    const Self = @This();

    pos_emb: []f32,
    x: []f32,
    o: []f32,
    logits: []f32,
    decoded: []u8,

    // Intermediate buffers.
    _h: []f32,
    _4xh: []f32,
    _qkv: []f32,
    _q: []f32,
    _k: []f32,
    _v: []f32,
    _attn: []f32,

    allocator: std.mem.Allocator,

    pub fn init(config: GPTConfig, allocator: std.mem.Allocator) !Self {
        return Self{
            .pos_emb = try allocator.alloc(f32, 1 * config.n_embed),
            .x = try allocator.alloc(f32, 1 * config.n_embed),
            .o = try allocator.alloc(f32, 1 * config.n_embed),
            .logits = try allocator.alloc(f32, config.vocab_size),
            .decoded = try allocator.alloc(u8, 20),

            ._h = try allocator.alloc(f32, 1 * config.n_embed),
            ._4xh = try allocator.alloc(f32, 1 * 4 * config.n_embed),
            ._qkv = try allocator.alloc(f32, 1 * 3 * config.n_embed),
            ._q = try allocator.alloc(f32, 1 * config.n_embed),
            ._k = try allocator.alloc(f32, config.context_size * config.n_embed),
            ._v = try allocator.alloc(f32, config.context_size * config.n_embed),
            ._attn = try allocator.alloc(f32, 1 * config.context_size),

            .allocator = allocator,
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
    pub fn forward(self: Self, inputs: []const f32, state: State) void {
        self.c_fc.forward(inputs, state._4xh);
        ops.gelu(state._4xh);
        self.c_proj.forward(state._4xh, state.o);
    }
};

const Block = struct {
    const Self = @This();

    n_embed: usize,
    ln_1: ops.LayerNorm,
    attn: ops.CausalSelfAttention,
    ln_2: ops.LayerNorm,
    mlp: MLP,
    k_cache: []f32,
    v_cache: []f32,

    pub fn init(
        n_embed: usize,
        ln_1: ops.LayerNorm,
        attn: ops.CausalSelfAttention,
        ln_2: ops.LayerNorm,
        mlp: MLP,
        k_cache: []f32,
        v_cache: []f32,
    ) Self {
        return Self{
            .n_embed = n_embed,
            .ln_1 = ln_1,
            .attn = attn,
            .ln_2 = ln_2,
            .mlp = mlp,
            .k_cache = k_cache,
            .v_cache = v_cache,
        };
    }

    /// Computes the forward pass and writes the result to both state.x and state.o. This
    /// enables sequentially calling multiple Block.forwards() in a row without having to copy
    /// memory.
    pub fn forward(self: Self, seq_len: usize, inputs: []const f32, state: State) void {
        // Create a copy of x for residual computation.
        @memcpy(state._h, inputs);

        self.ln_1.forward(state._h);
        self.attn.forward(
            seq_len,
            state._h,
            self.k_cache[0 .. seq_len * self.n_embed],
            self.v_cache[0 .. seq_len * self.n_embed],
            state.o,
            state._qkv,
            state._q,
            state._k[0 .. seq_len * self.n_embed],
            state._v[0 .. seq_len * self.n_embed],
            state._attn[0..seq_len],
        );
        for (0..state.o) |i| {
            state._h[i] = state.o[i] + inputs[i];
            state.x[i] = state._h[i];
        }
        self.ln_2.forward(state._h);
        self.mlp.forward(state._h, state);
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
    pub fn forward(self: Self, seq_len: usize, token: usize, state: State) void {
        self.wpe.forward(&[1]usize{seq_len - 1}, state.pos_emb);
        self.wte.forward(&[1]usize{token}, state.x);
        for (0..self.config.n_embed) |i| {
            state.x[i] += state.pos_emb[i];
        }

        // Forward the transformer.
        for (0..self.h.len) |i| {
            self.h[i].forward(seq_len, state.x, state);
        }
        self.ln_f.forward(state.x);

        // Compute logits.
        self.lm_head.forward(state.x, state.logits);
    }

    /// Samples the next token.
    pub fn sample(self: Self, seq_len: usize, temp: f32, token: usize, state: State) usize {
        self.forward(seq_len, token, state);
        for (0..state.logits.len) |i| {
            state.logits[i] /= temp;
        }
        ops.softmax(state.logits);
        var rng = std.rand.DefaultPrng.init(@intCast(std.time.timestamp()));
        var random = rng.random();
        return random.weightedIndex(f32, state.logits);
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
    const k_cache = try allocator.alloc(f32, config.context_size * config.n_embed);
    const v_cache = try allocator.alloc(f32, config.context_size * config.n_embed);

    return Block.init(config.n_embed, ln_1, attn, ln_2, mlp, k_cache, v_cache);
}

pub fn load_gpt(config: GPTConfig, allocator: std.mem.Allocator) !GPT {
    var wte = try load_embedding("wte", config.vocab_size, config.n_embed, allocator);
    const wpe = try load_embedding("wpe", config.context_size, config.n_embed, allocator);
    var h = try allocator.alloc(Block, config.n_layer);
    for (0..h.len) |i| {
        h[i] = try load_block(i, config, allocator);
    }
    const ln_f = try load_layer_norm("ln_f", config.n_embed, allocator);
    const lm_head = ops.Linear.init(config.n_embed, config.vocab_size, wte.weight, null);
    return GPT.init(config, wte, wpe, h, ln_f, lm_head);
}

pub fn load_encoder(allocator: std.mem.Allocator) !bpe.Encoder {
    const parsed_encoder = try ops.load_json("models/124M/encoder.json", allocator);
    const parsed_bytes_encoder = try ops.load_json("models/124M/byte_encoder.json", allocator);
    return bpe.Encoder.init(parsed_encoder.object, parsed_bytes_encoder.object, allocator);
}

pub fn generate(
    gpt: GPT,
    encoder: bpe.Encoder,
    temp: f32,
    inputs: []usize,
    state: State,
) void {
    var token: usize = undefined;
    for (0..gpt.config.context_size) |s| {
        if (s < inputs.len) {
            // Fill up KV cache.
            token = inputs[s];
            gpt.forward(s + 1, token, state);
        } else {
            // Generate.
            token = gpt.sample(s + 1, temp, token, state);
        }
        const decoded_len = encoder.decode(&[_]usize{token}, state.decoded);
        std.debug.print("{s}", .{state.decoded[0..decoded_len]});
    }
}

pub fn main() !void {
    const temp = 0.8;
    const config = GPTConfig.init(50257, 1024, 12, 12, 768);

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();
    const allocator = arena.allocator();

    var inputs = try allocator.alloc(usize, config.context_size);
    var encoder = try load_encoder(allocator);
    defer encoder.deinit();
    var state = try State.init(config, allocator);
    const gpt = try load_gpt(config, allocator);

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    const prompt = args[1];

    const input_tokens = encoder.encode(prompt, inputs);
    generate(
        gpt,
        encoder,
        temp,
        inputs[0..input_tokens],
        state,
    );
}
