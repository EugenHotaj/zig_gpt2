const std = @import("std");
const ops = @import("main.zig");

pub fn expectTensorsApproxEqual(expected: []const f32, actual: []const f32) !void {
    for (0..expected.len) |i| {
        try std.testing.expectApproxEqAbs(
            expected[i],
            actual[i],
            // TODO(eugenhotaj): Why are we losing precision? We're applying the same
            // operations (at least mathematically) as PyTorch. After 1e-6 all tests
            // start failing.
            1e-6,
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

test "GELU" {
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

test "Softmax" {
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
