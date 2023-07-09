const std = @import("std");

pub const Encoder = struct {
    const Self = @This();
    const idx_to_token_t = std.hash_map.AutoHashMap(usize, []const u8);

    token_to_idx: std.json.ObjectMap,
    idx_to_token: idx_to_token_t,
    unicode_to_byte: std.json.ObjectMap,

    pub fn init(
        token_to_idx: std.json.ObjectMap,
        unicode_to_byte: std.json.ObjectMap,
        allocator: std.mem.Allocator,
    ) !Self {
        var idx_to_token = idx_to_token_t.init(allocator);
        var it = token_to_idx.iterator();
        while (it.next()) |item| {
            try idx_to_token.put(@intCast(item.value_ptr.*.integer), item.key_ptr.*);
        }
        return Self{
            .token_to_idx = token_to_idx,
            .idx_to_token = idx_to_token,
            .unicode_to_byte = unicode_to_byte,
        };
    }

    pub fn deinit(self: *Self) void {
        self.idx_to_token.deinit();
    }

    pub fn decode(self: Self, inputs: []const usize, outputs: []u8) void {
        var outputs_len: usize = 0;
        for (inputs) |idx| {
            const token = self.idx_to_token.get(idx).?;
            var i: usize = 0;
            while (i < token.len) {
                var char: []const u8 = undefined;
                if (self.unicode_to_byte.contains(token[i .. i + 1])) {
                    char = token[i .. i + 1];
                    i += 1;
                } else {
                    char = token[i .. i + 2];
                    i += 2;
                }
                outputs[outputs_len] = @intCast(self.unicode_to_byte.get(char).?.integer);
                outputs_len += 1;
            }
        }
    }
};
