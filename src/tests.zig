const std = @import("std");
const ops = @import("ops.zig");

pub fn expectTensorsApproxEqual(expected: []const f32, actual: []const f32) !void {
    for (0..expected.len) |i| {
        try std.testing.expectApproxEqAbs(
            expected[i],
            actual[i],
            // TODO(eugenhotaj): Can we push the precision to 1e-7?
            5e-7,
        );
    }
}

test "Linear" {
    const batch_size = 3;
    const in_features = 5;
    const out_features = 10;

    const allocator = std.heap.page_allocator;
    const weight = try ops.load_tensor(
        "models/test/linear_weight",
        &[_]usize{ out_features, in_features },
        f32,
        &allocator,
    );
    defer allocator.free(weight);
    const bias = try ops.load_tensor(
        "models/test/linear_bias",
        &[_]usize{out_features},
        f32,
        &allocator,
    );
    defer allocator.free(bias);
    const inputs = try ops.load_tensor(
        "models/test/linear_inputs",
        &[_]usize{ batch_size, in_features },
        f32,
        &allocator,
    );
    defer allocator.free(inputs);
    const expected = try ops.load_tensor(
        "models/test/linear_outputs",
        &[_]usize{ batch_size, out_features },
        f32,
        &allocator,
    );
    defer allocator.free(expected);

    const linear = ops.Linear(in_features, out_features).init(weight, bias);
    const actual = try linear.forward(inputs, &allocator);
    defer allocator.free(actual);

    try expectTensorsApproxEqual(expected, actual);
}

test "Embedding" {
    const batch_size = 3;
    const vocab_size = 10;
    const embedding_dim = 5;

    const allocator = std.heap.page_allocator;
    const weight = try ops.load_tensor(
        "models/test/embedding_weight",
        &[_]usize{ vocab_size, embedding_dim },
        f32,
        &allocator,
    );
    defer allocator.free(weight);
    const inputs = try ops.load_tensor(
        "models/test/embedding_inputs",
        &[_]usize{ batch_size, 1 },
        usize,
        &allocator,
    );
    defer allocator.free(inputs);
    const expected = try ops.load_tensor(
        "models/test/embedding_outputs",
        &[_]usize{ batch_size, embedding_dim },
        f32,
        &allocator,
    );
    defer allocator.free(expected);

    const embedding = ops.Embedding(embedding_dim).init(weight);
    const actual = try embedding.forward(inputs, &allocator);
    defer allocator.free(actual);

    try expectTensorsApproxEqual(expected, actual);
}

test "LayerNorm" {
    const batch_size = 3;
    const in_features = 5;

    const allocator = std.heap.page_allocator;
    const weight = try ops.load_tensor(
        "models/test/layer_norm_weight",
        &[_]usize{ in_features, 1 },
        f32,
        &allocator,
    );
    defer allocator.free(weight);
    const bias = try ops.load_tensor(
        "models/test/layer_norm_bias",
        &[_]usize{in_features},
        f32,
        &allocator,
    );
    defer allocator.free(bias);
    const inputs = try ops.load_tensor(
        "models/test/layer_norm_inputs",
        &[_]usize{ batch_size, in_features },
        f32,
        &allocator,
    );
    defer allocator.free(inputs);
    const expected = try ops.load_tensor(
        "models/test/layer_norm_outputs",
        &[_]usize{ batch_size, in_features },
        f32,
        &allocator,
    );
    defer allocator.free(expected);

    const layer_norm = ops.LayerNorm(in_features).init(weight, bias);
    layer_norm.forward(inputs);
    const actual = inputs;

    try expectTensorsApproxEqual(expected, actual);
}

test "CausalSelfAttention.split_qkv" {
    const batch_size = 3;
    const n_heads = 3;
    const seq_len = 5;
    const head_dim = 4;
    const n_embed = n_heads * head_dim;

    const allocator = std.heap.page_allocator;
    var inputs = try ops.load_tensor(
        "models/test/split_inputs",
        &[_]usize{ batch_size, seq_len, 3 * n_embed },
        f32,
        &allocator,
    );
    defer allocator.free(inputs);
    var expected_q = try ops.load_tensor(
        "models/test/split_q",
        &[_]usize{ batch_size, seq_len, n_heads, head_dim },
        f32,
        &allocator,
    );
    defer allocator.free(expected_q);
    var expected_k = try ops.load_tensor(
        "models/test/split_k",
        &[_]usize{ batch_size, seq_len, n_heads, head_dim },
        f32,
        &allocator,
    );
    defer allocator.free(expected_k);
    var expected_v = try ops.load_tensor(
        "models/test/split_v",
        &[_]usize{ batch_size, seq_len, n_heads, head_dim },
        f32,
        &allocator,
    );
    defer allocator.free(expected_v);

    const actual_q = try ops.CausalSelfAttention(n_heads, seq_len, head_dim).split_qkv(inputs, 0, &allocator);
    const actual_k = try ops.CausalSelfAttention(n_heads, seq_len, head_dim).split_qkv(inputs, 1, &allocator);
    const actual_v = try ops.CausalSelfAttention(n_heads, seq_len, head_dim).split_qkv(inputs, 2, &allocator);
    defer allocator.free(actual_q);
    defer allocator.free(actual_k);
    defer allocator.free(actual_v);

    try expectTensorsApproxEqual(expected_q, actual_q);
    try expectTensorsApproxEqual(expected_k, actual_k);
    try expectTensorsApproxEqual(expected_v, actual_v);
}

test "CausalSelfAttention.transpose" {
    const batch_size = 3;
    const n_heads = 3;
    const seq_len = 5;
    const head_dim = 4;

    const allocator = std.heap.page_allocator;
    var inputs = try ops.load_tensor(
        "models/test/transpose_inputs",
        &[_]usize{ batch_size, seq_len, n_heads, head_dim },
        f32,
        &allocator,
    );
    defer allocator.free(inputs);
    var expected = try ops.load_tensor(
        "models/test/transpose_outputs",
        &[_]usize{ batch_size, n_heads, seq_len, head_dim },
        f32,
        &allocator,
    );
    defer allocator.free(expected);

    const actual = try ops.CausalSelfAttention(n_heads, seq_len, head_dim).transpose(inputs, &allocator);
    defer allocator.free(actual);

    try expectTensorsApproxEqual(expected, actual);
}

test "CausalSelfAttention.forward" {
    const batch_size = 3;
    const n_heads = 3;
    const seq_len = 5;
    const head_dim = 4;
    const n_embed = n_heads * head_dim;

    const allocator = std.heap.page_allocator;
    var inputs = try ops.load_tensor(
        "models/test/attn_inputs",
        &[_]usize{ batch_size, seq_len, n_embed },
        f32,
        &allocator,
    );
    defer allocator.free(inputs);
    var c_attn_weight = try ops.load_tensor(
        "models/test/attn_c_attn_weight",
        &[_]usize{ n_embed, 3 * n_embed },
        f32,
        &allocator,
    );
    defer allocator.free(c_attn_weight);
    var c_attn_bias = try ops.load_tensor(
        "models/test/attn_c_attn_bias",
        &[_]usize{3 * n_embed},
        f32,
        &allocator,
    );
    var c_proj_weight = try ops.load_tensor(
        "models/test/attn_c_proj_weight",
        &[_]usize{ n_embed, n_embed },
        f32,
        &allocator,
    );
    defer allocator.free(c_proj_weight);
    var c_proj_bias = try ops.load_tensor(
        "models/test/attn_c_proj_bias",
        &[_]usize{n_embed},
        f32,
        &allocator,
    );
    defer allocator.free(c_proj_bias);
    var expected = try ops.load_tensor(
        "models/test/attn_outputs",
        &[_]usize{ batch_size, seq_len, n_embed },
        f32,
        &allocator,
    );
    defer allocator.free(expected);

    const attn = ops.CausalSelfAttention(n_heads, seq_len, head_dim).init(
        c_attn_weight,
        c_attn_bias,
        c_proj_weight,
        c_proj_bias,
    );
    const actual = try attn.forward(inputs, &allocator);
    defer allocator.free(actual);

    try expectTensorsApproxEqual(expected, actual);
}

test "gelu" {
    const batch_size = 3;
    const in_features = 5;

    const allocator = std.heap.page_allocator;
    var inputs = try ops.load_tensor(
        "models/test/gelu_inputs",
        &[_]usize{ batch_size, in_features },
        f32,
        &allocator,
    );
    defer allocator.free(inputs);
    const expected = try ops.load_tensor(
        "models/test/gelu_outputs",
        &[_]usize{ batch_size, in_features },
        f32,
        &allocator,
    );
    defer allocator.free(expected);

    ops.gelu(inputs);
    const actual = inputs;

    try expectTensorsApproxEqual(expected, actual);
}

test "softmax" {
    const batch_size = 3;
    const in_features = 5;

    const allocator = std.heap.page_allocator;
    var inputs = try ops.load_tensor(
        "models/test/softmax_inputs",
        &[_]usize{ batch_size, in_features },
        f32,
        &allocator,
    );
    defer allocator.free(inputs);
    const expected = try ops.load_tensor(
        "models/test/softmax_outputs",
        &[_]usize{ batch_size, in_features },
        f32,
        &allocator,
    );
    defer allocator.free(expected);

    ops.softmax(in_features, inputs);
    const actual = inputs;

    try expectTensorsApproxEqual(expected, actual);
}

test "scaled_dot_product_attention" {
    const batch_size = 2;
    const n_heads = 3;
    const seq_len = 5;
    const head_dim = 4;

    const allocator = std.heap.page_allocator;
    const q = try ops.load_tensor(
        "models/test/sdpa_q",
        &[_]usize{ batch_size, n_heads, seq_len, head_dim },
        f32,
        &allocator,
    );
    defer allocator.free(q);
    const k = try ops.load_tensor(
        "models/test/sdpa_k",
        &[_]usize{ batch_size, n_heads, seq_len, head_dim },
        f32,
        &allocator,
    );
    defer allocator.free(k);
    const v = try ops.load_tensor(
        "models/test/sdpa_v",
        &[_]usize{ batch_size, n_heads, seq_len, head_dim },
        f32,
        &allocator,
    );
    defer allocator.free(v);

    const expected = try ops.load_tensor(
        "models/test/sdpa_outputs",
        &[_]usize{ batch_size, n_heads, seq_len, head_dim },
        f32,
        &allocator,
    );
    defer allocator.free(expected);

    const actual = try ops.scaled_dot_product_attention(q, k, v, n_heads, seq_len, head_dim, &allocator);
    defer allocator.free(actual);

    try expectTensorsApproxEqual(expected, actual);
}
