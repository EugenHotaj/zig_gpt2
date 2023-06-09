const std = @import("std");
const ops = @import("main.zig");

// TODO(eugenhotaj): Why are we losing precision? We're applying the same operation
// (at least mathematically) as PyTorch. After 1e-6 all tests start failing.
pub fn expectTensorsApproxEqual(expected: []const f32, actual: []const f32, tol: f32) !void {
    for (0..expected.len) |i| {
        try std.testing.expectApproxEqAbs(
            expected[i],
            actual[i],
            tol,
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
        &allocator,
    );
    defer allocator.free(weight);
    const bias = try ops.load_tensor(
        "models/test/linear_bias",
        &[_]usize{out_features},
        &allocator,
    );
    defer allocator.free(bias);
    const inputs = try ops.load_tensor(
        "models/test/linear_inputs",
        &[_]usize{ batch_size, in_features },
        &allocator,
    );
    defer allocator.free(inputs);
    const expected = try ops.load_tensor(
        "models/test/linear_outputs",
        &[_]usize{ batch_size, out_features },
        &allocator,
    );
    defer allocator.free(expected);

    const linear = ops.Linear(in_features, out_features).init(weight, bias);
    const actual = try linear.forward(inputs, &allocator);
    defer allocator.free(actual);

    try expectTensorsApproxEqual(expected, actual, 1e-6);
}

test "GELU" {
    const batch_size = 3;
    const in_features = 5;

    const allocator = std.heap.page_allocator;
    const inputs = try ops.load_tensor(
        "models/test/gelu_inputs",
        &[_]usize{ batch_size, in_features },
        &allocator,
    );
    defer allocator.free(inputs);
    const expected = try ops.load_tensor(
        "models/test/gelu_outputs",
        &[_]usize{ batch_size, in_features },
        &allocator,
    );
    defer allocator.free(expected);

    const actual = try ops.gelu(inputs, &allocator);

    try expectTensorsApproxEqual(expected, actual, 1e-6);
}
