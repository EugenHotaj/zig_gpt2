const std = @import("std");
const c = @cImport(@cInclude("regex.h"));

pub const Encoder = struct {
    const Self = @This();
    const idx_to_token_t = std.hash_map.AutoHashMap(usize, []const u8);

    token_to_idx: std.json.ObjectMap,
    idx_to_token: idx_to_token_t,
    unicode_to_byte: std.json.ObjectMap,
    regex: *c.regex_t,

    pub fn init(
        token_to_idx: std.json.ObjectMap,
        unicode_to_byte: std.json.ObjectMap,
        allocator: std.mem.Allocator,
    ) !Self {
        // Setup encoders.
        var idx_to_token = idx_to_token_t.init(allocator);
        var it = token_to_idx.iterator();
        while (it.next()) |item| {
            try idx_to_token.put(@intCast(item.value_ptr.*.integer), item.key_ptr.*);
        }

        // Setup regex.
        var slice = try allocator.alignedAlloc(u8, @alignOf(c.regex_t), @sizeOf(c.regex_t));
        const regex = @as(*c.regex_t, @ptrCast(slice.ptr));
        const contractions = "'s|'t|'re|'ve|'m|'ll|'d|";
        const letters = "[[:space:]]?[[:alpha:]]+|";
        const numbers = "[[:space:]]?[[:digit:]]+";
        const others = "|[[:space:]]?[^[:space:][:alpha:][:digit:]]+";
        // TODO(eugenhotaj): Multiple spaces between tokens are not handled correctly!
        const space = "|[[:space:]]+";
        _ = c.regcomp(regex, contractions ++ letters ++ numbers ++ others ++ space, c.REG_EXTENDED);

        return Self{
            .token_to_idx = token_to_idx,
            .idx_to_token = idx_to_token,
            .unicode_to_byte = unicode_to_byte,
            .regex = regex,
        };
    }

    pub fn deinit(self: *Self) void {
        self.idx_to_token.deinit();
        c.regfree(self.regex);
    }

    pub fn encode(self: Self, inputs: []const u8) void {
        var matches: [1]c.regmatch_t = undefined;
        var last_idx: usize = 0;
        while (last_idx < inputs.len) {
            _ = c.regexec(self.regex, inputs[last_idx..].ptr, matches.len, &matches, 0);
            const match = matches[0];
            const start_offset = last_idx + @as(usize, @intCast(match.rm_so));
            last_idx += @intCast(match.rm_eo);
            const out = inputs[start_offset..last_idx];
            std.debug.print("{s}\n", .{out});
        }
    }

    pub fn decode(self: Self, inputs: []const usize, outputs: []u8) usize {
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
        return outputs_len;
    }
};
