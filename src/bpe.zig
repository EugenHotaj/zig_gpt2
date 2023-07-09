const std = @import("std");

pub const Encoder = struct {
    const Self = @This();
    const decoder_t = std.hash_map.AutoHashMap(usize, []const u8);

    encoder: std.json.ObjectMap, // string->int
    decoder: decoder_t, // int->str
    bytes_encoder: std.json.ObjectMap, // string->int

    pub fn init(
        encoder: std.json.ObjectMap,
        bytes_encoder: std.json.ObjectMap,
        allocator: std.mem.Allocator,
    ) !Self {
        var decoder = decoder_t.init(allocator);
        var it = encoder.iterator();
        while (it.next()) |item| {
            try decoder.put(@intCast(item.value_ptr.*.integer), item.key_ptr.*);
        }
        return Self{ .encoder = encoder, .decoder = decoder, .bytes_encoder = bytes_encoder };
    }

    pub fn deinit(self: *Self) void {
        self.decoder.deinit();
    }

    pub fn decode(self: Self, inputs: []const usize, outputs: []u8) void {
        var outputs_len: usize = 0;
        for (inputs) |idx| {
            const token = self.decoder.get(idx).?;
            var i: usize = 0;
            while (i < token.len) {
                var char: []const u8 = undefined;
                if (self.bytes_encoder.contains(token[i .. i + 1])) {
                    char = token[i .. i + 1];
                    i += 1;
                } else {
                    char = token[i .. i + 2];
                    i += 2;
                }
                outputs[outputs_len] = @intCast(self.bytes_encoder.get(char).?.integer);
                outputs_len += 1;
            }
        }
    }
};
