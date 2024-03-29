const std = @import("std");
const ops = @import("ops.zig");

pub fn expectTensorsApproxEqual(expected: []const f32, actual: []const f32) !void {
    for (0..expected.len) |i| {
        if (@fabs(expected[i]) < 1e-3) {
            try std.testing.expectApproxEqAbs(
                expected[i],
                actual[i],
                5e-7,
            );
        } else {
            try std.testing.expectApproxEqRel(
                expected[i],
                actual[i],
                6e-4,
            );
        }
    }
}

test "Linear" {
    const batch_size = 3;
    const in_features = 768;
    const out_features = 4 * 768;

    const allocator = std.heap.page_allocator;
    const weight = try ops.load_tensor(
        "models/test/linear_weight",
        &[_]usize{ in_features, out_features },
        f32,
        allocator,
    );
    defer allocator.free(weight);
    const bias = try ops.load_tensor(
        "models/test/linear_bias",
        &[_]usize{out_features},
        f32,
        allocator,
    );
    defer allocator.free(bias);
    const inputs = try ops.load_tensor(
        "models/test/linear_inputs",
        &[_]usize{ batch_size, in_features },
        f32,
        allocator,
    );
    defer allocator.free(inputs);
    const expected = try ops.load_tensor(
        "models/test/linear_outputs",
        &[_]usize{ batch_size, out_features },
        f32,
        allocator,
    );
    defer allocator.free(expected);

    // Test Linear with bias.
    const linear = ops.Linear.init(in_features, out_features, weight, bias);
    const actual = try allocator.alloc(f32, batch_size * out_features);
    defer allocator.free(actual);
    linear.forward(inputs, actual);
    try expectTensorsApproxEqual(expected, actual);

    // Test Linear no bias.
    const expected_no_bias = try ops.load_tensor(
        "models/test/linear_outputs_no_bias",
        &[_]usize{ batch_size, out_features },
        f32,
        allocator,
    );
    defer allocator.free(expected_no_bias);

    const no_bias = ops.Linear.init(in_features, out_features, weight, null);
    const actual_no_bias = try allocator.alloc(f32, batch_size * out_features);
    defer allocator.free(actual_no_bias);
    no_bias.forward(inputs, actual_no_bias);
    try expectTensorsApproxEqual(expected_no_bias, actual_no_bias);
}

test "Embedding" {
    const batch_size = 3;
    const vocab_size = 10;
    const embedding_dim = 768;

    const allocator = std.heap.page_allocator;
    const weight = try ops.load_tensor(
        "models/test/embedding_weight",
        &[_]usize{ vocab_size, embedding_dim },
        f32,
        allocator,
    );
    defer allocator.free(weight);
    const inputs = try ops.load_tensor(
        "models/test/embedding_inputs",
        &[_]usize{batch_size},
        usize,
        allocator,
    );
    defer allocator.free(inputs);
    const expected = try ops.load_tensor(
        "models/test/embedding_outputs",
        &[_]usize{ batch_size, embedding_dim },
        f32,
        allocator,
    );
    defer allocator.free(expected);

    const embedding = ops.Embedding.init(embedding_dim, weight);
    const actual = try allocator.alloc(f32, batch_size * embedding_dim);
    defer allocator.free(actual);
    embedding.forward(inputs, actual);

    try expectTensorsApproxEqual(expected, actual);
}

test "LayerNorm" {
    const batch_size = 3;
    const in_features = 768;

    const allocator = std.heap.page_allocator;
    const weight = try ops.load_tensor(
        "models/test/layer_norm_weight",
        &[_]usize{in_features},
        f32,
        allocator,
    );
    defer allocator.free(weight);
    const bias = try ops.load_tensor(
        "models/test/layer_norm_bias",
        &[_]usize{in_features},
        f32,
        allocator,
    );
    defer allocator.free(bias);
    const inputs = try ops.load_tensor(
        "models/test/layer_norm_inputs",
        &[_]usize{ batch_size, in_features },
        f32,
        allocator,
    );
    defer allocator.free(inputs);
    const expected = try ops.load_tensor(
        "models/test/layer_norm_outputs",
        &[_]usize{ batch_size, in_features },
        f32,
        allocator,
    );
    defer allocator.free(expected);

    const layer_norm = ops.LayerNorm.init(in_features, weight, bias);
    layer_norm.forward(inputs);
    const actual = inputs;

    try expectTensorsApproxEqual(expected, actual);
}

test "CausalSelfAttention.split_qkv" {
    const batch_size = 3;
    const n_heads = 12;
    const seq_len = 5;
    const n_embed = 768;

    const allocator = std.heap.page_allocator;
    var inputs = try ops.load_tensor(
        "models/test/split_inputs",
        &[_]usize{ batch_size, seq_len, 3 * n_embed },
        f32,
        allocator,
    );
    defer allocator.free(inputs);
    var expected_q = try ops.load_tensor(
        "models/test/split_q",
        &[_]usize{ batch_size, seq_len, n_embed },
        f32,
        allocator,
    );
    defer allocator.free(expected_q);
    var expected_k = try ops.load_tensor(
        "models/test/split_k",
        &[_]usize{ batch_size, seq_len, n_embed },
        f32,
        allocator,
    );
    defer allocator.free(expected_k);
    var expected_v = try ops.load_tensor(
        "models/test/split_v",
        &[_]usize{ batch_size, seq_len, n_embed },
        f32,
        allocator,
    );
    defer allocator.free(expected_v);

    const fake_attn = ops.CausalSelfAttention.init(n_heads, n_embed, undefined, undefined);
    const actual_q = try allocator.alloc(f32, batch_size * seq_len * n_embed);
    defer allocator.free(actual_q);
    fake_attn.split_qkv(seq_len, inputs, 0, actual_q);

    const actual_k = try allocator.alloc(f32, batch_size * seq_len * n_embed);
    defer allocator.free(actual_k);
    fake_attn.split_qkv(seq_len, inputs, 1, actual_k);

    const actual_v = try allocator.alloc(f32, batch_size * seq_len * n_embed);
    defer allocator.free(actual_v);
    fake_attn.split_qkv(seq_len, inputs, 2, actual_v);

    try expectTensorsApproxEqual(expected_q, actual_q);
    try expectTensorsApproxEqual(expected_k, actual_k);
    try expectTensorsApproxEqual(expected_v, actual_v);
}

test "CausalSelfAttention.transpose" {
    const batch_size = 3;
    const n_heads = 12;
    const seq_len = 5;
    const n_embed = 768;
    const head_dim = n_embed / n_heads;

    const allocator = std.heap.page_allocator;
    var inputs = try ops.load_tensor(
        "models/test/transpose_inputs",
        &[_]usize{ batch_size, seq_len, n_heads, head_dim },
        f32,
        allocator,
    );
    defer allocator.free(inputs);
    var expected = try ops.load_tensor(
        "models/test/transpose_outputs",
        &[_]usize{ batch_size, n_heads, seq_len, head_dim },
        f32,
        allocator,
    );
    defer allocator.free(expected);

    const actual = try allocator.alloc(f32, batch_size * seq_len * n_embed);
    defer allocator.free(actual);
    ops.CausalSelfAttention.transpose(
        [3]usize{ seq_len, n_heads, head_dim },
        inputs,
        actual,
    );

    try expectTensorsApproxEqual(expected, actual);
}

test "CausalSelfAttention.forward" {
    const batch_size = 1;
    const seq_len = 5;
    const n_heads = 12;
    const head_dim = 64;
    const n_embed = n_heads * head_dim;

    const allocator = std.heap.page_allocator;
    var inputs = try ops.load_tensor(
        "models/test/attn_inputs",
        &[_]usize{ batch_size, seq_len, n_embed },
        f32,
        allocator,
    );
    defer allocator.free(inputs);
    var c_attn_weight = try ops.load_tensor(
        "models/test/attn_c_attn_weight",
        &[_]usize{ n_embed, 3 * n_embed },
        f32,
        allocator,
    );
    defer allocator.free(c_attn_weight);
    var c_attn_bias = try ops.load_tensor(
        "models/test/attn_c_attn_bias",
        &[_]usize{3 * n_embed},
        f32,
        allocator,
    );
    var c_proj_weight = try ops.load_tensor(
        "models/test/attn_c_proj_weight",
        &[_]usize{ n_embed, n_embed },
        f32,
        allocator,
    );
    defer allocator.free(c_proj_weight);
    var c_proj_bias = try ops.load_tensor(
        "models/test/attn_c_proj_bias",
        &[_]usize{n_embed},
        f32,
        allocator,
    );
    defer allocator.free(c_proj_bias);
    var expected = try ops.load_tensor(
        "models/test/attn_outputs",
        &[_]usize{ batch_size, seq_len, n_embed },
        f32,
        allocator,
    );
    defer allocator.free(expected);

    const c_attn = ops.Linear.init(n_embed, 3 * n_embed, c_attn_weight, c_attn_bias);
    const c_proj = ops.Linear.init(n_embed, n_embed, c_proj_weight, c_proj_bias);
    const attn = ops.CausalSelfAttention.init(n_heads, n_embed, c_attn, c_proj);

    const actual = try allocator.alloc(f32, batch_size * seq_len * n_embed);
    defer allocator.free(actual);
    const k_cache = try allocator.alloc(f32, batch_size * seq_len * n_embed);
    defer allocator.free(k_cache);
    const v_cache = try allocator.alloc(f32, batch_size * seq_len * n_embed);
    defer allocator.free(v_cache);
    const _qkv = try allocator.alloc(f32, batch_size * 1 * 3 * n_embed);
    defer allocator.free(_qkv);
    const _q = try allocator.alloc(f32, batch_size * 1 * n_embed);
    defer allocator.free(_q);
    const _k = try allocator.alloc(f32, batch_size * seq_len * n_embed);
    defer allocator.free(_k);
    const _v = try allocator.alloc(f32, batch_size * seq_len * n_embed);
    defer allocator.free(_v);
    const _attn = try allocator.alloc(f32, 1 * seq_len);
    defer allocator.free(_attn);

    for (0..seq_len) |s| {
        attn.forward(
            s + 1,
            inputs[s * n_embed .. (s + 1) * n_embed],
            k_cache[0 .. (s + 1) * n_embed],
            v_cache[0 .. (s + 1) * n_embed],
            actual[s * n_embed .. (s + 1) * n_embed],
            _qkv,
            _q,
            _k[0 .. (s + 1) * n_embed],
            _v[0 .. (s + 1) * n_embed],
            _attn[0..(s + 1)],
        );
        try expectTensorsApproxEqual(
            expected[s * n_embed .. (s + 1) * n_embed],
            actual[s * n_embed .. (s + 1) * n_embed],
        );
    }
}

test "gelu" {
    const batch_size = 3;
    const in_features = 768;

    const allocator = std.heap.page_allocator;
    var inputs = try ops.load_tensor(
        "models/test/gelu_inputs",
        &[_]usize{ batch_size, in_features },
        f32,
        allocator,
    );
    defer allocator.free(inputs);
    const expected = try ops.load_tensor(
        "models/test/gelu_outputs",
        &[_]usize{ batch_size, in_features },
        f32,
        allocator,
    );
    defer allocator.free(expected);

    ops.gelu(inputs);
    const actual = inputs;

    try expectTensorsApproxEqual(expected, actual);
}

test "softmax" {
    const batch_size = 3;
    const in_features = 768;

    const allocator = std.heap.page_allocator;
    var inputs = try ops.load_tensor(
        "models/test/softmax_inputs",
        &[_]usize{ batch_size, in_features },
        f32,
        allocator,
    );
    defer allocator.free(inputs);
    const expected = try ops.load_tensor(
        "models/test/softmax_outputs",
        &[_]usize{ batch_size, in_features },
        f32,
        allocator,
    );
    defer allocator.free(expected);

    for (0..batch_size) |b| {
        ops.softmax(inputs[b * in_features .. (b + 1) * in_features]);
    }
    const actual = inputs;

    try expectTensorsApproxEqual(expected, actual);
}
