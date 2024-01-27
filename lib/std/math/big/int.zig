const std = @import("../../std.zig");
const builtin = @import("builtin");
const math = std.math;
const Limb = std.math.big.Limb;
const limb_bits = @typeInfo(Limb).Int.bits;
const HalfLimb = std.math.big.HalfLimb;
const half_limb_bits = @typeInfo(HalfLimb).Int.bits;
const DoubleLimb = std.math.big.DoubleLimb;
const SignedDoubleLimb = std.math.big.SignedDoubleLimb;
const Log2Limb = std.math.big.Log2Limb;
const Allocator = std.mem.Allocator;
const mem = std.mem;
const maxInt = std.math.maxInt;
const minInt = std.math.minInt;
const assert = std.debug.assert;
const Endian = std.builtin.Endian;
const Signedness = std.builtin.Signedness;
const native_endian = builtin.cpu.arch.endian();

const debug_safety = false;

/// Returns the number of limbs needed to store `scalar`, which must be a
/// primitive integer value.
/// Note: A comptime-known upper bound of this value that may be used
/// instead if `scalar` is not already comptime-known is
/// `calcTwosCompLimbCount(@typeInfo(@TypeOf(scalar)).Int.bits)`
pub fn calcLimbLen(scalar: anytype) usize {
    if (scalar == 0) {
        return 1;
    }

    const w_value = @abs(scalar);
    return @as(usize, @intCast(@divFloor(@as(Limb, @intCast(math.log2(w_value))), limb_bits) + 1));
}

pub fn calcToStringLimbsBufferLen(a_len: usize, base: u8) usize {
    if (math.isPowerOfTwo(base))
        return 0;
    return a_len + 2 + a_len + calcDivLimbsBufferLen(a_len, 1);
}

pub fn calcDivLimbsBufferLen(a_len: usize, b_len: usize) usize {
    return a_len + b_len + 4;
}

pub fn calcMulLimbsBufferLen(a_len: usize, b_len: usize, aliases: usize) usize {
    return aliases * @max(a_len, b_len);
}

pub fn calcMulWrapLimbsBufferLen(bit_count: usize, a_len: usize, b_len: usize, aliases: usize) usize {
    const req_limbs = calcTwosCompLimbCount(bit_count);
    return aliases * @min(req_limbs, @max(a_len, b_len));
}

pub fn calcSetStringLimbsBufferLen(base: u8, string_len: usize) usize {
    const limb_count = calcSetStringLimbCount(base, string_len);
    return calcMulLimbsBufferLen(limb_count, limb_count, 2);
}

pub fn calcSetStringLimbCount(base: u8, string_len: usize) usize {
    return (string_len + (limb_bits / base - 1)) / (limb_bits / base);
}

pub fn calcPowLimbsBufferLen(a_bit_count: usize, y: usize) usize {
    // The 2 accounts for the minimum space requirement for llmulacc
    return 2 + (a_bit_count * y + (limb_bits - 1)) / limb_bits;
}

pub fn calcSqrtLimbsBufferLen(a_bit_count: usize) usize {
    const a_limb_count = (a_bit_count - 1) / limb_bits + 1;
    const shift = (a_bit_count + 1) / 2;
    const u_s_rem_limb_count = 1 + ((shift / limb_bits) + 1);
    return a_limb_count + 3 * u_s_rem_limb_count + calcDivLimbsBufferLen(a_limb_count, u_s_rem_limb_count);
}

// Compute the number of limbs required to store a 2s-complement number of `bit_count` bits.
pub fn calcTwosCompLimbCount(bit_count: usize) usize {
    return std.math.divCeil(usize, bit_count, @bitSizeOf(Limb)) catch unreachable;
}

/// Used to indicate either limit of a 2s-complement integer.
pub const TwosCompIntLimit = enum {
    // The low limit, either 0x00 (unsigned) or (-)0x80 (signed) for an 8-bit integer.
    min,

    // The high limit, either 0xFF (unsigned) or 0x7F (signed) for an 8-bit integer.
    max,
};

pub const Sign = enum(u1) {
    pos,
    neg,

    pub fn negate(s: Sign) Sign {
        return @enumFromInt(@intFromEnum(s) ^ 1);
    }

    // Returns a sign given a bool which is true if positive, false if negative
    pub fn fromBool(is_positive: bool) Sign {
        return @enumFromInt(@intFromBool(!is_positive));
    }
};

pub const Metadata = enum(isize) {
    zero = 0,
    _,

    pub fn init(new_sign: Sign, new_len: usize) Metadata {
        const signed_len: isize = @intCast(new_len);
        return @enumFromInt(if (new_sign == .pos) signed_len else -signed_len);
    }

    pub fn len(m: Metadata) usize {
        return @intCast(@abs(@as(isize, @intFromEnum(m))));
    }

    pub fn sign(m: Metadata) Sign {
        return @enumFromInt(@intFromEnum(m) >> (@bitSizeOf(Metadata) - 1));
    }

    pub fn withLen(m: Metadata, new_len: usize) Metadata {
        return init(m.sign(), new_len);
    }

    pub fn withSign(m: Metadata, new_sign: Sign) Metadata {
        return init(new_sign, m.len());
    }
};

/// A arbitrary-precision big integer, with a fixed set of mutable limbs.
pub const Mutable = struct {
    /// Raw digits. These are:
    ///
    /// * Little-endian ordered
    /// * limbs.len >= 1
    /// * Zero is represented as TODO
    ///
    /// Accessing limbs directly should be avoided.
    /// These are allocated limbs; the `len` field tells the valid range.
    limbs: []Limb,
    metadata: Metadata,

    /// Returns the number of limbs used to represent the current value.
    pub fn len(self: Mutable) usize {
        return self.metadata.len();
    }

    /// Returns the sign of the current value.
    pub fn sign(self: Mutable) Sign {
        return self.metadata.sign();
    }

    pub fn toConst(self: Mutable) Const {
        return .{
            .limbs = self.limbs.ptr,
            .metadata = self.metadata,
        };
    }

    /// Returns true if `a == 0`.
    pub fn eqlZero(self: Mutable) bool {
        return self.toConst().eqlZero();
    }

    /// Asserts that the allocator owns the limbs memory. If this is not the case,
    /// use `toConst().toManaged()`.
    pub fn toManaged(self: Mutable, allocator: Allocator) Managed {
        return .{
            .allocator = allocator,
            .limbs = self.limbs,
            .metadata = self.metadata,
        };
    }

    /// `value` is a primitive integer type.
    /// Asserts the value fits within the provided `limbs_buffer`.
    /// Note: `calcLimbLen` can be used to figure out how big an array to allocate for `limbs_buffer`.
    pub fn init(limbs_buffer: []Limb, value: anytype) Mutable {
        limbs_buffer[0] = 0;
        var self: Mutable = .{
            .limbs = limbs_buffer,
            .metadata = Metadata.init(.pos, 1),
        };
        self.set(value);
        return self;
    }

    /// Copies the value of a Const to an existing Mutable so that they both have the same value.
    /// Asserts the value fits in the limbs buffer.
    pub fn copy(self: *Mutable, other: Const) void {
        if (self.limbs.ptr != other.limbs) {
            @memcpy(self.limbs[0..other.len()], other.limbs[0..other.len()]);
        }
        self.metadata = other.metadata;
    }

    /// Efficiently swap an Mutable with another. This swaps the limb pointers and a full copy is not
    /// performed. The address of the limbs field will not be the same after this function.
    pub fn swap(self: *Mutable, other: *Mutable) void {
        mem.swap(Mutable, self, other);
    }

    pub fn dump(self: Mutable) void {
        for (self.limbs[0..self.len()]) |limb| {
            std.debug.print("{x} ", .{limb});
        }
        std.debug.print("capacity={} positive={}\n", .{ self.limbs.len, self.sign() == .pos });
    }

    /// Clones an Mutable and returns a new Mutable with the same value. The new Mutable is a deep copy and
    /// can be modified separately from the original.
    /// Asserts that limbs is big enough to store the value.
    pub fn clone(other: Mutable, limbs: []Limb) Mutable {
        @memcpy(limbs[0..other.len], other.limbs[0..other.len]);
        return .{
            .limbs = limbs,
            .len = other.len,
            .positive = other.positive,
        };
    }

    pub fn setLen(self: *Mutable, new_len: usize) void {
        self.metadata = self.metadata.withLen(new_len);
    }

    pub fn setSign(self: *Mutable, new_sign: Sign) void {
        self.metadata = self.metadata.withSign(new_sign);
    }

    pub fn negate(self: *Mutable) void {
        self.setSign(self.sign().negate());
    }

    /// Modify to become the absolute value
    pub fn abs(self: *Mutable) void {
        self.setSign(.pos);
    }

    /// Sets the Mutable to value. Value must be an primitive integer type.
    /// Asserts the value fits within the limbs buffer.
    /// Note: `calcLimbLen` can be used to figure out how big the limbs buffer
    /// needs to be to store a specific value.
    pub fn set(self: *Mutable, value: anytype) void {
        const T = @TypeOf(value);
        const needed_limbs = calcLimbLen(value);
        assert(needed_limbs <= self.limbs.len); // value too big

        self.metadata = Metadata.init(Sign.fromBool(value >= 0), needed_limbs);

        switch (@typeInfo(T)) {
            .Int => |info| {
                var w_value = @abs(value);

                if (info.bits <= limb_bits) {
                    self.limbs[0] = w_value;
                } else {
                    var i: usize = 0;
                    while (true) : (i += 1) {
                        self.limbs[i] = @as(Limb, @truncate(w_value));
                        w_value >>= limb_bits;

                        if (w_value == 0) break;
                    }
                }
            },
            .ComptimeInt => {
                comptime var w_value = @abs(value);

                if (w_value <= maxInt(Limb)) {
                    self.limbs[0] = w_value;
                } else {
                    const mask = (1 << limb_bits) - 1;

                    comptime var i = 0;
                    inline while (true) : (i += 1) {
                        self.limbs[i] = w_value & mask;
                        w_value >>= limb_bits;

                        if (w_value == 0) break;
                    }
                }
            },
            else => @compileError("cannot set Mutable using type " ++ @typeName(T)),
        }
    }

    /// Set self from the string representation `value`.
    ///
    /// `value` must contain only digits <= `base` and is case insensitive.  Base prefixes are
    /// not allowed (e.g. 0x43 should simply be 43).  Underscores in the input string are
    /// ignored and can be used as digit separators.
    ///
    /// Asserts there is enough memory for the value in `self.limbs`. An upper bound on number of limbs can
    /// be determined with `calcSetStringLimbCount`.
    /// Asserts the base is in the range [2, 16].
    ///
    /// Returns an error if the value has invalid digits for the requested base.
    ///
    /// `limbs_buffer` is used for temporary storage. The size required can be found with
    /// `calcSetStringLimbsBufferLen`.
    ///
    /// If `allocator` is provided, it will be used for temporary storage to improve
    /// multiplication performance. `error.OutOfMemory` is handled with a fallback algorithm.
    pub fn setString(
        self: *Mutable,
        base: u8,
        value: []const u8,
        limbs_buffer: []Limb,
        allocator: ?Allocator,
    ) error{InvalidCharacter}!void {
        assert(base >= 2 and base <= 16);

        var i: usize = 0;
        var positive = true;
        if (value.len > 0 and value[0] == '-') {
            positive = false;
            i += 1;
        }

        const ap_base: Const = .{ .limbs = &[_]Limb{base}, .metadata = Metadata.init(.pos, 1) };
        self.set(0);

        for (value[i..]) |ch| {
            if (ch == '_') {
                continue;
            }
            const d = try std.fmt.charToDigit(ch, base);
            const ap_d: Const = .{ .limbs = &[_]Limb{d}, .metadata = Metadata.init(.pos, 1) };

            self.mul(self.toConst(), ap_base, limbs_buffer, allocator);
            self.add(self.toConst(), ap_d);
        }
        self.setSign(Sign.fromBool(positive));
    }

    /// Set self to either bound of a 2s-complement integer.
    /// Note: The result is still sign-magnitude, not twos complement! In order to convert the
    /// result to twos complement, it is sufficient to take the absolute value.
    ///
    /// Asserts the result fits in `r`. An upper bound on the number of limbs needed by
    /// r is `calcTwosCompLimbCount(bit_count)`.
    pub fn setTwosCompIntLimit(
        r: *Mutable,
        limit: TwosCompIntLimit,
        signedness: Signedness,
        bit_count: usize,
    ) void {
        // Handle zero-bit types.
        if (bit_count == 0) {
            r.set(0);
            return;
        }

        const req_limbs = calcTwosCompLimbCount(bit_count);
        const bit = @as(Log2Limb, @truncate(bit_count - 1));
        const signmask = @as(Limb, 1) << bit; // 0b0..010..0 where 1 is the sign bit.
        const mask = (signmask << 1) -% 1; // 0b0..011..1 where the leftmost 1 is the sign bit.

        r.setSign(.pos);

        switch (signedness) {
            .signed => switch (limit) {
                .min => {
                    // Negative bound, signed = -0x80.
                    @memset(r.limbs[0 .. req_limbs - 1], 0);
                    r.limbs[req_limbs - 1] = signmask;
                    r.metadata = Metadata.init(.neg, req_limbs);
                },
                .max => {
                    // Positive bound, signed = 0x7F
                    // Note, in this branch we need to normalize because the first bit is
                    // supposed to be 0.

                    // Special case for 1-bit integers.
                    if (bit_count == 1) {
                        r.set(0);
                    } else {
                        const new_req_limbs = calcTwosCompLimbCount(bit_count - 1);
                        const msb = @as(Log2Limb, @truncate(bit_count - 2));
                        const new_signmask = @as(Limb, 1) << msb; // 0b0..010..0 where 1 is the sign bit.
                        const new_mask = (new_signmask << 1) -% 1; // 0b0..001..1 where the rightmost 0 is the sign bit.

                        @memset(r.limbs[0 .. new_req_limbs - 1], maxInt(Limb));
                        r.limbs[new_req_limbs - 1] = new_mask;
                        r.setLen(new_req_limbs);
                    }
                },
            },
            .unsigned => switch (limit) {
                .min => {
                    // Min bound, unsigned = 0x00
                    r.set(0);
                },
                .max => {
                    // Max bound, unsigned = 0xFF
                    @memset(r.limbs[0 .. req_limbs - 1], maxInt(Limb));
                    r.limbs[req_limbs - 1] = mask;
                    r.setLen(req_limbs);
                },
            },
        }
    }

    /// r = a + scalar
    ///
    /// r and a may be aliases.
    /// scalar is a primitive integer type.
    ///
    /// Asserts the result fits in `r`. An upper bound on the number of limbs needed by
    /// r is `@max(a.limbs.len, calcLimbLen(scalar)) + 1`.
    pub fn addScalar(r: *Mutable, a: Const, scalar: anytype) void {
        // Normally we could just determine the number of limbs needed with calcLimbLen,
        // but that is not comptime-known when scalar is not a comptime_int.  Instead, we
        // use calcTwosCompLimbCount for a non-comptime_int scalar, which can be pessimistic
        // in the case that scalar happens to be small in magnitude within its type, but it
        // is well worth being able to use the stack and not needing an allocator passed in.
        // Note that Mutable.init still sets len to calcLimbLen(scalar) in any case.
        const limb_len = comptime switch (@typeInfo(@TypeOf(scalar))) {
            .ComptimeInt => calcLimbLen(scalar),
            .Int => |info| calcTwosCompLimbCount(info.bits),
            else => @compileError("expected scalar to be an int"),
        };
        var limbs: [limb_len]Limb = undefined;
        const operand = init(&limbs, scalar).toConst();
        return add(r, a, operand);
    }

    /// Base implementation for addition. Adds `@max(a.limbs.len, b.limbs.len)` elements from a and b,
    /// and returns whether any overflow occurred.
    /// r, a and b may be aliases.
    ///
    /// Asserts r has enough elements to hold the result. The upper bound is `@max(a.limbs.len, b.limbs.len)`.
    fn addCarry(r: *Mutable, a: Const, b: Const) bool {
        if (a.eqlZero()) {
            r.copy(b);
            return false;
        } else if (b.eqlZero()) {
            r.copy(a);
            return false;
        } else if (a.sign() != b.sign()) {
            if (a.sign() == .pos) {
                // (a) + (-b) => a - b
                return r.subCarry(a, b.abs());
            } else {
                // (-a) + (b) => b - a
                return r.subCarry(b, a.abs());
            }
        } else {
            r.setSign(a.sign());
            if (a.len() >= b.len()) {
                const c = lladdcarry(r.limbs, a.limbs[0..a.len()], b.limbs[0..b.len()]);
                r.normalize(a.len());
                return c != 0;
            } else {
                const c = lladdcarry(r.limbs, b.limbs[0..b.len()], a.limbs[0..a.len()]);
                r.normalize(b.len());
                return c != 0;
            }
        }
    }

    /// r = a + b
    ///
    /// r, a and b may be aliases.
    ///
    /// Asserts the result fits in `r`. An upper bound on the number of limbs needed by
    /// r is `@max(a.limbs.len, b.limbs.len) + 1`.
    pub fn add(r: *Mutable, a: Const, b: Const) void {
        if (r.addCarry(a, b)) {
            // Fix up the result. Note that addCarry normalizes by a.limbs.len or b.limbs.len,
            // so we need to set the length here.
            const msl = @max(a.len(), b.len());
            // `[add|sub]Carry` normalizes by `msl`, so we need to fix up the result manually here.
            // Note, the fact that it normalized means that the intermediary limbs are zero here.
            r.setLen(msl + 1);
            r.limbs[msl] = 1; // If this panics, there wasn't enough space in `r`.
        }
    }

    /// r = a + b with 2s-complement wrapping semantics. Returns whether overflow occurred.
    /// r, a and b may be aliases
    ///
    /// Asserts the result fits in `r`. An upper bound on the number of limbs needed by
    /// r is `calcTwosCompLimbCount(bit_count)`.
    pub fn addWrap(r: *Mutable, a: Const, b: Const, signedness: Signedness, bit_count: usize) bool {
        const req_limbs = calcTwosCompLimbCount(bit_count);

        // Slice of the upper bits if they exist, these will be ignored and allows us to use addCarry to determine
        // if an overflow occurred.
        const x = Const{
            .limbs = a.limbs,
            .metadata = Metadata.init(a.sign(), @min(req_limbs, a.len())),
        };

        const y = Const{
            .limbs = b.limbs,
            .metadata = Metadata.init(b.sign(), @min(req_limbs, b.len())),
        };

        var carry_truncated = false;
        if (r.addCarry(x, y)) {
            // There are two possibilities here:
            // - We overflowed req_limbs. In this case, the carry is ignored, as it would be removed by
            //   truncate anyway.
            // - a and b had less elements than req_limbs, and those were overflowed. This case needs to be handled.
            //   Note: after this we still might need to wrap.
            const msl = @max(a.len(), b.len());
            if (msl < req_limbs) {
                r.limbs[msl] = 1;
                r.setLen(req_limbs);
                @memset(r.limbs[msl + 1 .. req_limbs], 0);
            } else {
                carry_truncated = true;
            }
        }

        if (!r.toConst().fitsInTwosComp(signedness, bit_count)) {
            r.truncate(r.toConst(), signedness, bit_count);
            return true;
        }

        return carry_truncated;
    }

    /// r = a + b with 2s-complement saturating semantics.
    /// r, a and b may be aliases.
    ///
    /// Assets the result fits in `r`. Upper bound on the number of limbs needed by
    /// r is `calcTwosCompLimbCount(bit_count)`.
    pub fn addSat(r: *Mutable, a: Const, b: Const, signedness: Signedness, bit_count: usize) void {
        const req_limbs = calcTwosCompLimbCount(bit_count);

        // Slice of the upper bits if they exist, these will be ignored and allows us to use addCarry to determine
        // if an overflow occurred.
        const x = Const{
            .limbs = a.limbs,
            .metadata = Metadata.init(a.sign(), @min(req_limbs, a.len())),
        };

        const y = Const{
            .limbs = b.limbs,
            .metadata = Metadata.init(b.sign(), @min(req_limbs, b.len())),
        };

        if (r.addCarry(x, y)) {
            // There are two possibilities here:
            // - We overflowed req_limbs, in which case we need to saturate.
            // - a and b had less elements than req_limbs, and those were overflowed.
            //   Note: In this case, might _also_ need to saturate.
            const msl = @max(a.len(), b.len());
            if (msl < req_limbs) {
                r.limbs[msl] = 1;
                r.setLen(req_limbs);
                // Note: Saturation may still be required if msl == req_limbs - 1
            } else {
                // Overflowed req_limbs, definitely saturate.
                r.setTwosCompIntLimit(if (r.sign() == .pos) .max else .min, signedness, bit_count);
            }
        }

        // Saturate if the result didn't fit.
        r.saturate(r.toConst(), signedness, bit_count);
    }

    /// Base implementation for subtraction. Subtracts `@max(a.limbs.len, b.limbs.len)` elements from a and b,
    /// and returns whether any overflow occurred.
    /// r, a and b may be aliases.
    ///
    /// Asserts r has enough elements to hold the result. The upper bound is `@max(a.limbs.len, b.limbs.len)`.
    fn subCarry(r: *Mutable, a: Const, b: Const) bool {
        if (a.eqlZero()) {
            r.copy(b);
            r.setSign(b.sign().negate());
            return false;
        } else if (b.eqlZero()) {
            r.copy(a);
            return false;
        } else if (a.sign() != b.sign()) {
            if (a.sign() == .pos) {
                // (a) - (-b) => a + b
                return r.addCarry(a, b.abs());
            } else {
                // (-a) - (b) => -a + -b
                return r.addCarry(a, b.negate());
            }
        } else if (a.sign() == .pos) {
            if (a.order(b) != .lt) {
                // (a) - (b) => a - b
                const c = llsubcarry(r.limbs, a.limbs[0..a.len()], b.limbs[0..b.len()]);
                r.normalize(a.len());
                r.setSign(.pos);
                return c != 0;
            } else {
                // (a) - (b) => -b + a => -(b - a)
                const c = llsubcarry(r.limbs, b.limbs[0..b.len()], a.limbs[0..a.len()]);
                r.normalize(b.len());
                r.setSign(.neg);
                return c != 0;
            }
        } else {
            if (a.order(b) == .lt) {
                // (-a) - (-b) => -(a - b)
                const c = llsubcarry(r.limbs, a.limbs[0..a.len()], b.limbs[0..b.len()]);
                r.normalize(a.len());
                r.setSign(.neg);
                return c != 0;
            } else {
                // (-a) - (-b) => --b + -a => b - a
                const c = llsubcarry(r.limbs, b.limbs[0..b.len()], a.limbs[0..a.len()]);
                r.normalize(b.len());
                r.setSign(.pos);
                return c != 0;
            }
        }
    }

    /// r = a - b
    ///
    /// r, a and b may be aliases.
    ///
    /// Asserts the result fits in `r`. An upper bound on the number of limbs needed by
    /// r is `@max(a.limbs.len, b.limbs.len) + 1`. The +1 is not needed if both operands are positive.
    pub fn sub(r: *Mutable, a: Const, b: Const) void {
        r.add(a, b.negate());
    }

    /// r = a - b with 2s-complement wrapping semantics. Returns whether any overflow occurred.
    ///
    /// r, a and b may be aliases
    /// Asserts the result fits in `r`. An upper bound on the number of limbs needed by
    /// r is `calcTwosCompLimbCount(bit_count)`.
    pub fn subWrap(r: *Mutable, a: Const, b: Const, signedness: Signedness, bit_count: usize) bool {
        return r.addWrap(a, b.negate(), signedness, bit_count);
    }

    /// r = a - b with 2s-complement saturating semantics.
    /// r, a and b may be aliases.
    ///
    /// Assets the result fits in `r`. Upper bound on the number of limbs needed by
    /// r is `calcTwosCompLimbCount(bit_count)`.
    pub fn subSat(r: *Mutable, a: Const, b: Const, signedness: Signedness, bit_count: usize) void {
        r.addSat(a, b.negate(), signedness, bit_count);
    }

    /// rma = a * b
    ///
    /// `rma` may alias with `a` or `b`.
    /// `a` and `b` may alias with each other.
    ///
    /// Asserts the result fits in `rma`. An upper bound on the number of limbs needed by
    /// rma is given by `a.limbs.len + b.limbs.len`.
    ///
    /// `limbs_buffer` is used for temporary storage. The amount required is given by `calcMulLimbsBufferLen`.
    pub fn mul(rma: *Mutable, a: Const, b: Const, limbs_buffer: []Limb, allocator: ?Allocator) void {
        var buf_index: usize = 0;

        const a_copy = if (rma.limbs.ptr == a.limbs) blk: {
            const start = buf_index;
            @memcpy(limbs_buffer[buf_index..][0..a.len()], a.limbs[0..a.len()]);
            buf_index += a.len();
            break :blk a.toMutable(limbs_buffer[start..buf_index]).toConst();
        } else a;

        const b_copy = if (rma.limbs.ptr == b.limbs) blk: {
            const start = buf_index;
            @memcpy(limbs_buffer[buf_index..][0..b.len()], b.limbs[0..b.len()]);
            buf_index += b.len();
            break :blk b.toMutable(limbs_buffer[start..buf_index]).toConst();
        } else b;

        return rma.mulNoAlias(a_copy, b_copy, allocator);
    }

    /// rma = a * b
    ///
    /// `rma` may not alias with `a` or `b`.
    /// `a` and `b` may alias with each other.
    ///
    /// Asserts the result fits in `rma`. An upper bound on the number of limbs needed by
    /// rma is given by `a.limbs.len + b.limbs.len`.
    ///
    /// If `allocator` is provided, it will be used for temporary storage to improve
    /// multiplication performance. `error.OutOfMemory` is handled with a fallback algorithm.
    pub fn mulNoAlias(rma: *Mutable, a: Const, b: Const, allocator: ?Allocator) void {
        assert(rma.limbs.ptr != a.limbs); // illegal aliasing
        assert(rma.limbs.ptr != b.limbs); // illegal aliasing

        if (a.len() == 1 and b.len() == 1) {
            const ov = @mulWithOverflow(a.limbs[0], b.limbs[0]);
            rma.limbs[0] = ov[0];
            if (ov[1] == 0) {
                rma.metadata = Metadata.init(Sign.fromBool(a.sign() == b.sign()), 1);
                return;
            }
        }

        @memset(rma.limbs[0 .. a.len() + b.len()], 0);

        llmulacc(.add, allocator, rma.limbs, a.limbs[0..a.len()], b.limbs[0..b.len()]);

        rma.normalize(a.len() + b.len());
        rma.setSign(Sign.fromBool(a.sign() == b.sign()));
    }

    /// rma = a * b with 2s-complement wrapping semantics.
    ///
    /// `rma` may alias with `a` or `b`.
    /// `a` and `b` may alias with each other.
    ///
    /// Asserts the result fits in `rma`. An upper bound on the number of limbs needed by
    /// rma is given by `a.limbs.len + b.limbs.len`.
    ///
    /// `limbs_buffer` is used for temporary storage. The amount required is given by `calcMulWrapLimbsBufferLen`.
    pub fn mulWrap(
        rma: *Mutable,
        a: Const,
        b: Const,
        signedness: Signedness,
        bit_count: usize,
        limbs_buffer: []Limb,
        allocator: ?Allocator,
    ) void {
        var buf_index: usize = 0;
        const req_limbs = calcTwosCompLimbCount(bit_count);

        const a_copy = if (rma.limbs.ptr == a.limbs) blk: {
            const start = buf_index;
            const a_len = @min(req_limbs, a.len());
            @memcpy(limbs_buffer[buf_index..][0..a_len], a.limbs[0..a_len]);
            buf_index += a_len;
            break :blk a.toMutable(limbs_buffer[start..buf_index]).toConst();
        } else a;

        const b_copy = if (rma.limbs.ptr == b.limbs) blk: {
            const start = buf_index;
            const b_len = @min(req_limbs, b.len());
            @memcpy(limbs_buffer[buf_index..][0..b_len], b.limbs[0..b_len]);
            buf_index += b_len;
            break :blk a.toMutable(limbs_buffer[start..buf_index]).toConst();
        } else b;

        return rma.mulWrapNoAlias(a_copy, b_copy, signedness, bit_count, allocator);
    }

    /// rma = a * b with 2s-complement wrapping semantics.
    ///
    /// `rma` may not alias with `a` or `b`.
    /// `a` and `b` may alias with each other.
    ///
    /// Asserts the result fits in `rma`. An upper bound on the number of limbs needed by
    /// rma is given by `a.limbs.len + b.limbs.len`.
    ///
    /// If `allocator` is provided, it will be used for temporary storage to improve
    /// multiplication performance. `error.OutOfMemory` is handled with a fallback algorithm.
    pub fn mulWrapNoAlias(
        rma: *Mutable,
        a: Const,
        b: Const,
        signedness: Signedness,
        bit_count: usize,
        allocator: ?Allocator,
    ) void {
        assert(rma.limbs.ptr != a.limbs); // illegal aliasing
        assert(rma.limbs.ptr != b.limbs); // illegal aliasing

        const req_limbs = calcTwosCompLimbCount(bit_count);

        // We can ignore the upper bits here, those results will be discarded anyway.
        const a_limbs = a.limbs[0..@min(req_limbs, a.len())];
        const b_limbs = b.limbs[0..@min(req_limbs, b.len())];

        @memset(rma.limbs[0..req_limbs], 0);

        llmulacc(.add, allocator, rma.limbs, a_limbs[0..a.len()], b_limbs[0..b.len()]);
        rma.normalize(@min(req_limbs, a.len() + b.len()));
        rma.setSign(Sign.fromBool(a.sign() == b.sign()));
        rma.truncate(rma.toConst(), signedness, bit_count);
    }

    /// r = @bitReverse(a) with 2s-complement semantics.
    /// r and a may be aliases.
    ///
    /// Asserts the result fits in `r`. Upper bound on the number of limbs needed by
    /// r is `calcTwosCompLimbCount(bit_count)`.
    pub fn bitReverse(r: *Mutable, a: Const, signedness: Signedness, bit_count: usize) void {
        if (bit_count == 0) return;

        r.copy(a);

        const limbs_required = calcTwosCompLimbCount(bit_count);

        if (a.sign() == .neg) {
            r.setSign(.pos); // Negate.
            r.bitNotWrap(r.toConst(), .unsigned, bit_count); // Bitwise NOT.
            r.addScalar(r.toConst(), 1); // Add one.
        } else if (limbs_required > a.len()) {
            // Zero-extend to our output length
            for (r.limbs[a.len()..limbs_required]) |*limb| {
                limb.* = 0;
            }
            r.setLen(limbs_required);
        }

        // 0b0..01..1000 with @log2(@sizeOf(Limb)) consecutive ones
        const endian_mask: usize = (@sizeOf(Limb) - 1) << 3;

        const bytes = std.mem.sliceAsBytes(r.limbs);
        var bits = std.packed_int_array.PackedIntSliceEndian(u1, .little).init(bytes, limbs_required * @bitSizeOf(Limb));

        var k: usize = 0;
        while (k < ((bit_count + 1) / 2)) : (k += 1) {
            var i = k;
            var rev_i = bit_count - i - 1;

            // This "endian mask" remaps a low (LE) byte to the corresponding high
            // (BE) byte in the Limb, without changing which limbs we are indexing
            if (native_endian == .big) {
                i ^= endian_mask;
                rev_i ^= endian_mask;
            }

            const bit_i = bits.get(i);
            const bit_rev_i = bits.get(rev_i);
            bits.set(i, bit_rev_i);
            bits.set(rev_i, bit_i);
        }

        // Calculate signed-magnitude representation for output
        if (signedness == .signed) {
            const last_bit = switch (native_endian) {
                .little => bits.get(bit_count - 1),
                .big => bits.get((bit_count - 1) ^ endian_mask),
            };
            if (last_bit == 1) {
                r.bitNotWrap(r.toConst(), .unsigned, bit_count); // Bitwise NOT.
                r.addScalar(r.toConst(), 1); // Add one.
                r.setSign(.neg); // Negate.
            }
        }
        r.normalize(r.len());
    }

    /// r = @byteSwap(a) with 2s-complement semantics.
    /// r and a may be aliases.
    ///
    /// Asserts the result fits in `r`. Upper bound on the number of limbs needed by
    /// r is `calcTwosCompLimbCount(8*byte_count)`.
    pub fn byteSwap(r: *Mutable, a: Const, signedness: Signedness, byte_count: usize) void {
        if (byte_count == 0) return;

        r.copy(a);
        const limbs_required = calcTwosCompLimbCount(8 * byte_count);

        if (a.sign() == .neg) {
            r.setSign(.pos); // Negate.
            r.bitNotWrap(r.toConst(), .unsigned, 8 * byte_count); // Bitwise NOT.
            r.addScalar(r.toConst(), 1); // Add one.
        } else if (limbs_required > a.len()) {
            // Zero-extend to our output length
            for (r.limbs[a.len()..limbs_required]) |*limb| {
                limb.* = 0;
            }
            r.setLen(limbs_required);
        }

        // 0b0..01..1 with @log2(@sizeOf(Limb)) trailing ones
        const endian_mask: usize = @sizeOf(Limb) - 1;

        var bytes = std.mem.sliceAsBytes(r.limbs);
        assert(bytes.len >= byte_count);

        var k: usize = 0;
        while (k < (byte_count + 1) / 2) : (k += 1) {
            var i = k;
            var rev_i = byte_count - k - 1;

            // This "endian mask" remaps a low (LE) byte to the corresponding high
            // (BE) byte in the Limb, without changing which limbs we are indexing
            if (native_endian == .big) {
                i ^= endian_mask;
                rev_i ^= endian_mask;
            }

            const byte_i = bytes[i];
            const byte_rev_i = bytes[rev_i];
            bytes[rev_i] = byte_i;
            bytes[i] = byte_rev_i;
        }

        // Calculate signed-magnitude representation for output
        if (signedness == .signed) {
            const last_byte = switch (native_endian) {
                .little => bytes[byte_count - 1],
                .big => bytes[(byte_count - 1) ^ endian_mask],
            };

            if (last_byte & (1 << 7) != 0) { // Check sign bit of last byte
                r.bitNotWrap(r.toConst(), .unsigned, 8 * byte_count); // Bitwise NOT.
                r.addScalar(r.toConst(), 1); // Add one.
                r.setSign(.neg); // Negate.
            }
        }
        r.normalize(r.len());
    }

    /// r = @popCount(a) with 2s-complement semantics.
    /// r and a may be aliases.
    ///
    /// Assets the result fits in `r`. Upper bound on the number of limbs needed by
    /// r is `calcTwosCompLimbCount(bit_count)`.
    pub fn popCount(r: *Mutable, a: Const, bit_count: usize) void {
        r.copy(a);

        if (a.sign() == .neg) {
            r.setSign(.pos); // Negate.
            r.bitNotWrap(r.toConst(), .unsigned, bit_count); // Bitwise NOT.
            r.addScalar(r.toConst(), 1); // Add one.
        }

        var sum: Limb = 0;
        for (r.limbs[0..r.len()]) |limb| {
            sum += @popCount(limb);
        }
        r.set(sum);
    }

    /// rma = a * a
    ///
    /// `rma` may not alias with `a`.
    ///
    /// Asserts the result fits in `rma`. An upper bound on the number of limbs needed by
    /// rma is given by `2 * a.limbs.len + 1`.
    ///
    /// If `allocator` is provided, it will be used for temporary storage to improve
    /// multiplication performance. `error.OutOfMemory` is handled with a fallback algorithm.
    pub fn sqrNoAlias(rma: *Mutable, a: Const, opt_allocator: ?Allocator) void {
        _ = opt_allocator;
        assert(rma.limbs.ptr != a.limbs); // illegal aliasing

        @memset(rma.limbs, 0);

        llsquareBasecase(rma.limbs, a.limbs[0..a.len()]);

        rma.normalize(2 * a.len() + 1);
        rma.setSign(.pos);
    }

    /// q = a / b (rem r)
    ///
    /// a / b are floored (rounded towards 0).
    /// q may alias with a or b.
    ///
    /// Asserts there is enough memory to store q and r.
    /// The upper bound for r limb count is `b.limbs.len`.
    /// The upper bound for q limb count is given by `a.limbs`.
    ///
    /// `limbs_buffer` is used for temporary storage. The amount required is given by `calcDivLimbsBufferLen`.
    pub fn divFloor(
        q: *Mutable,
        r: *Mutable,
        a: Const,
        b: Const,
        limbs_buffer: []Limb,
    ) void {
        const sep = a.len() + 2;
        var x = a.toMutable(limbs_buffer[0..sep]);
        var y = b.toMutable(limbs_buffer[sep..]);

        div(q, r, &x, &y);

        // Note, `div` performs truncating division, which satisfies
        // @divTrunc(a, b) * b + @rem(a, b) = a
        // so r = a - @divTrunc(a, b) * b
        // Note,  @rem(a, -b) = @rem(-b, a) = -@rem(a, b) = -@rem(-a, -b)
        // For divTrunc, we want to perform
        // @divFloor(a, b) * b + @mod(a, b) = a
        // Note:
        // @divFloor(-a, b)
        // = @divFloor(a, -b)
        // = -@divCeil(a, b)
        // = -@divFloor(a + b - 1, b)
        // = -@divTrunc(a + b - 1, b)

        // Note (1):
        // @divTrunc(a + b - 1, b) * b + @rem(a + b - 1, b) = a + b - 1
        // = @divTrunc(a + b - 1, b) * b + @rem(a - 1, b) = a + b - 1
        // = @divTrunc(a + b - 1, b) * b + @rem(a - 1, b) - b + 1 = a

        if (a.sign() == .pos and b.sign() == .pos) {
            // Positive-positive case, don't need to do anything.
        } else if (a.sign() == .pos and b.sign() == .neg) {
            // a/-b -> q is negative, and so we need to fix flooring.
            // Subtract one to make the division flooring.

            // @divFloor(a, -b) * -b + @mod(a, -b) = a
            // If b divides a exactly, we have @divFloor(a, -b) * -b = a
            // Else, we have @divFloor(a, -b) * -b > a, so @mod(a, -b) becomes negative

            // We have:
            // @divFloor(a, -b) * -b + @mod(a, -b) = a
            // = -@divTrunc(a + b - 1, b) * -b + @mod(a, -b) = a
            // = @divTrunc(a + b - 1, b) * b + @mod(a, -b) = a

            // Substitute a for (1):
            // @divTrunc(a + b - 1, b) * b + @rem(a - 1, b) - b + 1 = @divTrunc(a + b - 1, b) * b + @mod(a, -b)
            // Yields:
            // @mod(a, -b) = @rem(a - 1, b) - b + 1
            // Note that `r` holds @rem(a, b) at this point.
            //
            // If @rem(a, b) is not 0:
            //   @rem(a - 1, b) = @rem(a, b) - 1
            //   => @mod(a, -b) = @rem(a, b) - 1 - b + 1 = @rem(a, b) - b
            // Else:
            //   @rem(a - 1, b) = @rem(a + b - 1, b) = @rem(b - 1, b) = b - 1
            //   => @mod(a, -b) = b - 1 - b + 1 = 0
            if (!r.eqlZero()) {
                q.addScalar(q.toConst(), -1);
                r.setSign(.pos);
                r.sub(r.toConst(), y.toConst().abs());
            }
        } else if (a.sign() == .neg and b.sign() == .pos) {
            // -a/b -> q is negative, and so we need to fix flooring.
            // Subtract one to make the division flooring.

            // @divFloor(-a, b) * b + @mod(-a, b) = a
            // If b divides a exactly, we have @divFloor(-a, b) * b = -a
            // Else, we have @divFloor(-a, b) * b < -a, so @mod(-a, b) becomes positive

            // We have:
            // @divFloor(-a, b) * b + @mod(-a, b) = -a
            // = -@divTrunc(a + b - 1, b) * b + @mod(-a, b) = -a
            // = @divTrunc(a + b - 1, b) * b - @mod(-a, b) = a

            // Substitute a for (1):
            // @divTrunc(a + b - 1, b) * b + @rem(a - 1, b) - b + 1 = @divTrunc(a + b - 1, b) * b - @mod(-a, b)
            // Yields:
            // @rem(a - 1, b) - b + 1 = -@mod(-a, b)
            // => -@mod(-a, b) = @rem(a - 1, b) - b + 1
            // => @mod(-a, b) = -(@rem(a - 1, b) - b + 1) = -@rem(a - 1, b) + b - 1
            //
            // If @rem(a, b) is not 0:
            //   @rem(a - 1, b) = @rem(a, b) - 1
            //   => @mod(-a, b) = -(@rem(a, b) - 1) + b - 1 = -@rem(a, b) + 1 + b - 1 = -@rem(a, b) + b
            // Else :
            //   @rem(a - 1, b) = b - 1
            //   => @mod(-a, b) = -(b - 1) + b - 1 = 0
            if (!r.eqlZero()) {
                q.addScalar(q.toConst(), -1);
                r.setSign(.neg);
                r.add(r.toConst(), y.toConst().abs());
            }
        } else if (a.sign() == .neg and b.sign() == .neg) {
            // a/b -> q is positive, don't need to do anything to fix flooring.

            // @divFloor(-a, -b) * -b + @mod(-a, -b) = -a
            // If b divides a exactly, we have @divFloor(-a, -b) * -b = -a
            // Else, we have @divFloor(-a, -b) * -b > -a, so @mod(-a, -b) becomes negative

            // We have:
            // @divFloor(-a, -b) * -b + @mod(-a, -b) = -a
            // = @divTrunc(a, b) * -b + @mod(-a, -b) = -a
            // = @divTrunc(a, b) * b - @mod(-a, -b) = a

            // We also have:
            // @divTrunc(a, b) * b + @rem(a, b) = a

            // Substitute a:
            // @divTrunc(a, b) * b + @rem(a, b) = @divTrunc(a, b) * b - @mod(-a, -b)
            // => @rem(a, b) = -@mod(-a, -b)
            // => @mod(-a, -b) = -@rem(a, b)
            r.setSign(.neg);
        }
    }

    /// q = a / b (rem r)
    ///
    /// a / b are truncated (rounded towards -inf).
    /// q may alias with a or b.
    ///
    /// Asserts there is enough memory to store q and r.
    /// The upper bound for r limb count is `b.limbs.len`.
    /// The upper bound for q limb count is given by `a.limbs.len`.
    ///
    /// `limbs_buffer` is used for temporary storage. The amount required is given by `calcDivLimbsBufferLen`.
    pub fn divTrunc(
        q: *Mutable,
        r: *Mutable,
        a: Const,
        b: Const,
        limbs_buffer: []Limb,
    ) void {
        const sep = a.len() + 2;
        var x = a.toMutable(limbs_buffer[0..sep]);
        var y = b.toMutable(limbs_buffer[sep..]);

        div(q, r, &x, &y);
    }

    /// r = a << shift, in other words, r = a * 2^shift
    ///
    /// r and a may alias.
    ///
    /// Asserts there is enough memory to fit the result. The upper bound Limb count is
    /// `a.limbs.len + (shift / (@sizeOf(Limb) * 8))`.
    pub fn shiftLeft(r: *Mutable, a: Const, shift: usize) void {
        llshl(r.limbs[0..], a.limbs[0..a.len()], shift);
        r.normalize(a.len() + (shift / limb_bits) + 1);
        r.setSign(a.sign());
    }

    /// r = a <<| shift with 2s-complement saturating semantics.
    ///
    /// r and a may alias.
    ///
    /// Asserts there is enough memory to fit the result. The upper bound Limb count is
    /// r is `calcTwosCompLimbCount(bit_count)`.
    pub fn shiftLeftSat(r: *Mutable, a: Const, shift: usize, signedness: Signedness, bit_count: usize) void {
        // Special case: When the argument is negative, but the result is supposed to be unsigned,
        // return 0 in all cases.
        if (a.sign() == .neg and signedness == .unsigned) {
            r.set(0);
            return;
        }

        // Check whether the shift is going to overflow. This is the case
        // when (in 2s complement) any bit above `bit_count - shift` is set in the unshifted value.
        // Note, the sign bit is not counted here.

        // Handle shifts larger than the target type. This also deals with
        // 0-bit integers.
        if (bit_count <= shift) {
            // In this case, there is only no overflow if `a` is zero.
            if (a.eqlZero()) {
                r.set(0);
            } else {
                r.setTwosCompIntLimit(if (a.sign() == .pos) .max else .min, signedness, bit_count);
            }
            return;
        }

        const checkbit = bit_count - shift - @intFromBool(signedness == .signed);
        // If `checkbit` and more significant bits are zero, no overflow will take place.

        if (checkbit >= a.len() * limb_bits) {
            // `checkbit` is outside the range of a, so definitely no overflow will take place. We
            // can defer to a normal shift.
            // Note that if `a` is normalized (which we assume), this checks for set bits in the upper limbs.

            // Note, in this case r should already have enough limbs required to perform the normal shift.
            // In this case the shift of the most significant limb may still overflow.
            r.shiftLeft(a, shift);
            return;
        } else if (checkbit < (a.len() - 1) * limb_bits) {
            // `checkbit` is not in the most significant limb. If `a` is normalized the most significant
            // limb will not be zero, so in this case we need to saturate. Note that `a.limbs.len` must be
            // at least one according to normalization rules.

            r.setTwosCompIntLimit(if (a.sign() == .pos) .max else .min, signedness, bit_count);
            return;
        }

        // Generate a mask with the bits to check in the most significant limb. We'll need to check
        // all bits with equal or more significance than checkbit.
        // const msb = @truncate(Log2Limb, checkbit);
        // const checkmask = (@as(Limb, 1) << msb) -% 1;

        if (a.limbs[a.len() - 1] >> @as(Log2Limb, @truncate(checkbit)) != 0) {
            // Need to saturate.
            r.setTwosCompIntLimit(if (a.sign() == .pos) .max else .min, signedness, bit_count);
            return;
        }

        // This shift should not be able to overflow, so invoke llshl and normalize manually
        // to avoid the extra required limb.
        llshl(r.limbs[0..], a.limbs[0..a.len()], shift);
        r.normalize(a.len() + (shift / limb_bits));
        r.setSign(a.sign());
    }

    /// r = a >> shift
    /// r and a may alias.
    ///
    /// Asserts there is enough memory to fit the result. The upper bound Limb count is
    /// `a.limbs.len - (shift / (@sizeOf(Limb) * 8))`.
    pub fn shiftRight(r: *Mutable, a: Const, shift: usize) void {
        if (a.len() <= shift / limb_bits) {
            // Shifting negative numbers converges to -1 instead of 0
            if (a.sign() == .pos) {
                r.metadata = Metadata.init(.pos, 1);
                r.limbs[0] = 0;
            } else {
                r.metadata = Metadata.init(.neg, 1);
                r.limbs[0] = 1;
            }
            return;
        }

        llshr(r.limbs[0..], a.limbs[0..a.len()], shift);
        r.normalize(a.len() - (shift / limb_bits));
        r.setSign(a.sign());
        // Shifting negative numbers converges to -1 instead of 0
        if (r.sign() == .neg and r.len() == 1 and r.limbs[0] == 0) {
            r.limbs[0] = 1;
        }
    }

    /// r = ~a under 2s complement wrapping semantics.
    /// r may alias with a.
    ///
    /// Assets that r has enough limbs to store the result. The upper bound Limb count is
    /// r is `calcTwosCompLimbCount(bit_count)`.
    pub fn bitNotWrap(r: *Mutable, a: Const, signedness: Signedness, bit_count: usize) void {
        r.copy(a.negate());
        const one: []const Limb = &.{1};
        const negative_one = Const{ .limbs = one.ptr, .metadata = Metadata.init(.neg, 1) };
        _ = r.addWrap(r.toConst(), negative_one, signedness, bit_count);
    }

    /// r = a | b under 2s complement semantics.
    /// r may alias with a or b.
    ///
    /// a and b are zero-extended to the longer of a or b.
    ///
    /// Asserts that r has enough limbs to store the result. Upper bound is `@max(a.limbs.len, b.limbs.len)`.
    pub fn bitOr(r: *Mutable, a: Const, b: Const) void {
        // Trivial cases, llsignedor does not support zero.
        if (a.eqlZero()) {
            r.copy(b);
            return;
        } else if (b.eqlZero()) {
            r.copy(a);
            return;
        }

        if (a.len() >= b.len()) {
            r.setSign(Sign.fromBool(llsignedor(r.limbs, a.limbs[0..a.len()], a.sign() == .pos, b.limbs[0..b.len()], b.sign() == .pos)));
            r.normalize(if (b.sign() == .pos) a.len() else b.len());
        } else {
            r.setSign(Sign.fromBool(llsignedor(r.limbs, b.limbs[0..b.len()], b.sign() == .pos, a.limbs[0..a.len()], a.sign() == .pos)));
            r.normalize(if (a.sign() == .pos) b.len() else a.len());
        }
    }

    /// r = a & b under 2s complement semantics.
    /// r may alias with a or b.
    ///
    /// Asserts that r has enough limbs to store the result.
    /// If a or b is positive, the upper bound is `@min(a.limbs.len, b.limbs.len)`.
    /// If a and b are negative, the upper bound is `@max(a.limbs.len, b.limbs.len) + 1`.
    pub fn bitAnd(r: *Mutable, a: Const, b: Const) void {
        // Trivial cases, llsignedand does not support zero.
        if (a.eqlZero()) {
            r.copy(a);
            return;
        } else if (b.eqlZero()) {
            r.copy(b);
            return;
        }

        if (a.len() >= b.len()) {
            r.setSign(Sign.fromBool(llsignedand(r.limbs, a.limbs[0..a.len()], a.sign() == .pos, b.limbs[0..b.len()], b.sign() == .pos)));
            r.normalize(if (a.sign() == .pos or b.sign() == .pos) b.len() else a.len() + 1);
        } else {
            r.setSign(Sign.fromBool(llsignedand(r.limbs, b.limbs[0..b.len()], b.sign() == .pos, a.limbs[0..a.len()], a.sign() == .pos)));
            r.normalize(if (a.sign() == .pos or b.sign() == .pos) a.len() else b.len() + 1);
        }
    }

    /// r = a ^ b under 2s complement semantics.
    /// r may alias with a or b.
    ///
    /// Asserts that r has enough limbs to store the result. If a and b share the same signedness, the
    /// upper bound is `@max(a.limbs.len, b.limbs.len)`. Otherwise, if either a or b is negative
    /// but not both, the upper bound is `@max(a.limbs.len, b.limbs.len) + 1`.
    pub fn bitXor(r: *Mutable, a: Const, b: Const) void {
        // Trivial cases, because llsignedxor does not support negative zero.
        if (a.eqlZero()) {
            r.copy(b);
            return;
        } else if (b.eqlZero()) {
            r.copy(a);
            return;
        }

        if (a.len() > b.len()) {
            r.setSign(Sign.fromBool(llsignedxor(r.limbs, a.limbs[0..a.len()], a.sign() == .pos, b.limbs[0..b.len()], b.sign() == .pos)));
            r.normalize(a.len() + @intFromBool(a.sign() != b.sign()));
        } else {
            r.setSign(Sign.fromBool(llsignedxor(r.limbs, b.limbs[0..b.len()], b.sign() == .pos, a.limbs[0..a.len()], a.sign() == .pos)));
            r.normalize(b.len() + @intFromBool(a.sign() != b.sign()));
        }
    }

    /// rma may alias x or y.
    /// x and y may alias each other.
    /// Asserts that `rma` has enough limbs to store the result. Upper bound is
    /// `@min(x.limbs.len, y.limbs.len)`.
    ///
    /// `limbs_buffer` is used for temporary storage during the operation. When this function returns,
    /// it will have the same length as it had when the function was called.
    pub fn gcd(rma: *Mutable, x: Const, y: Const, limbs_buffer: *std.ArrayList(Limb)) !void {
        const prev_len = limbs_buffer.items.len;
        defer limbs_buffer.shrinkRetainingCapacity(prev_len);
        const x_copy = if (rma.limbs.ptr == x.limbs) blk: {
            const start = limbs_buffer.items.len;
            try limbs_buffer.appendSlice(x.limbs[0..x.len()]);
            break :blk x.toMutable(limbs_buffer.items[start..]).toConst();
        } else x;
        const y_copy = if (rma.limbs.ptr == y.limbs) blk: {
            const start = limbs_buffer.items.len;
            try limbs_buffer.appendSlice(y.limbs[0..y.len()]);
            break :blk y.toMutable(limbs_buffer.items[start..]).toConst();
        } else y;

        return gcdLehmer(rma, x_copy, y_copy, limbs_buffer);
    }

    /// q = a ^ b
    ///
    /// r may not alias a.
    ///
    /// Asserts that `r` has enough limbs to store the result. Upper bound is
    /// `calcPowLimbsBufferLen(a.bitCountAbs(), b)`.
    ///
    /// `limbs_buffer` is used for temporary storage.
    /// The amount required is given by `calcPowLimbsBufferLen`.
    pub fn pow(r: *Mutable, a: Const, b: u32, limbs_buffer: []Limb) !void {
        assert(r.limbs.ptr != a.limbs.ptr); // illegal aliasing

        // Handle all the trivial cases first
        switch (b) {
            0 => {
                // a^0 = 1
                return r.set(1);
            },
            1 => {
                // a^1 = a
                return r.copy(a);
            },
            else => {},
        }

        if (a.eqlZero()) {
            // 0^b = 0
            return r.set(0);
        } else if (a.len() == 1 and a.limbs[0] == 1) {
            // 1^b = 1 and -1^b = ±1
            r.set(1);
            r.setSign(Sign.fromBool(a.sign() == .pos or (b & 1) == 0));
            return;
        }

        // Here a>1 and b>1
        const needed_limbs = calcPowLimbsBufferLen(a.bitCountAbs(), b);
        assert(r.limbs.len >= needed_limbs);
        assert(limbs_buffer.len >= needed_limbs);

        llpow(r.limbs, a.limbs[0..a.len()], b, limbs_buffer);

        r.normalize(needed_limbs);
        r.setSign(Sign.fromBool(a.sign() == .pos or (b & 1) == 0));
    }

    /// r = ⌊√a⌋
    ///
    /// r may alias a.
    ///
    /// Asserts that `r` has enough limbs to store the result. Upper bound is
    /// `(a.limbs.len - 1) / 2 + 1`.
    ///
    /// `limbs_buffer` is used for temporary storage.
    /// The amount required is given by `calcSqrtLimbsBufferLen`.
    pub fn sqrt(
        r: *Mutable,
        a: Const,
        limbs_buffer: []Limb,
    ) void {
        // Brent and Zimmermann, Modern Computer Arithmetic, Algorithm 1.13 SqrtInt
        // https://members.loria.fr/PZimmermann/mca/pub226.html
        var buf_index: usize = 0;
        var t = b: {
            const start = buf_index;
            buf_index += a.len();
            break :b Mutable.init(limbs_buffer[start..buf_index], 0);
        };
        var u = b: {
            const start = buf_index;
            const shift = (a.bitCountAbs() + 1) / 2;
            buf_index += 1 + ((shift / limb_bits) + 1);
            var m = Mutable.init(limbs_buffer[start..buf_index], 1);
            m.shiftLeft(m.toConst(), shift); // u must be >= ⌊√a⌋, and should be as small as possible for efficiency
            break :b m;
        };
        var s = b: {
            const start = buf_index;
            buf_index += u.limbs.len;
            break :b u.toConst().toMutable(limbs_buffer[start..buf_index]);
        };
        var rem = b: {
            const start = buf_index;
            buf_index += s.limbs.len;
            break :b Mutable.init(limbs_buffer[start..buf_index], 0);
        };

        while (true) {
            t.divFloor(&rem, a, s.toConst(), limbs_buffer[buf_index..]);
            t.add(t.toConst(), s.toConst());
            u.shiftRight(t.toConst(), 1);

            if (u.toConst().order(s.toConst()).compare(.gte)) {
                r.copy(s.toConst());
                return;
            }

            // Avoid copying u to s by swapping u and s
            const tmp_s = s;
            s = u;
            u = tmp_s;
        }
    }

    /// rma may not alias x or y.
    /// x and y may alias each other.
    /// Asserts that `rma` has enough limbs to store the result. Upper bound is given by `calcGcdNoAliasLimbLen`.
    ///
    /// `limbs_buffer` is used for temporary storage during the operation.
    pub fn gcdNoAlias(rma: *Mutable, x: Const, y: Const, limbs_buffer: *std.ArrayList(Limb)) !void {
        assert(rma.limbs.ptr != x.limbs); // illegal aliasing
        assert(rma.limbs.ptr != y.limbs); // illegal aliasing
        return gcdLehmer(rma, x, y, limbs_buffer);
    }

    fn gcdLehmer(result: *Mutable, xa: Const, ya: Const, limbs_buffer: *std.ArrayList(Limb)) !void {
        var x = try xa.toManaged(limbs_buffer.allocator);
        defer x.deinit();
        x.abs();

        var y = try ya.toManaged(limbs_buffer.allocator);
        defer y.deinit();
        y.abs();

        if (x.toConst().order(y.toConst()) == .lt) {
            x.swap(&y);
        }

        var t_big = try Managed.init(limbs_buffer.allocator);
        defer t_big.deinit();

        var r = try Managed.init(limbs_buffer.allocator);
        defer r.deinit();

        var tmp_x = try Managed.init(limbs_buffer.allocator);
        defer tmp_x.deinit();

        while (y.len() > 1 and !y.eqlZero()) {
            assert(x.sign() == .pos and y.sign() == .pos);
            assert(x.len() >= y.len());

            var xh: SignedDoubleLimb = x.limbs[x.len() - 1];
            var yh: SignedDoubleLimb = if (x.len() > y.len()) 0 else y.limbs[x.len() - 1];

            var A: SignedDoubleLimb = 1;
            var B: SignedDoubleLimb = 0;
            var C: SignedDoubleLimb = 0;
            var D: SignedDoubleLimb = 1;

            while (yh + C != 0 and yh + D != 0) {
                const q = @divFloor(xh + A, yh + C);
                const qp = @divFloor(xh + B, yh + D);
                if (q != qp) {
                    break;
                }

                var t = A - q * C;
                A = C;
                C = t;
                t = B - q * D;
                B = D;
                D = t;

                t = xh - q * yh;
                xh = yh;
                yh = t;
            }

            if (B == 0) {
                // t_big = x % y, r is unused
                try r.divTrunc(&t_big, &x, &y);
                assert(t_big.sign() == .pos);

                x.swap(&y);
                y.swap(&t_big);
            } else {
                var storage: [8]Limb = undefined;
                const Ap = fixedIntFromSignedDoubleLimb(A, storage[0..2]).toManaged(limbs_buffer.allocator);
                const Bp = fixedIntFromSignedDoubleLimb(B, storage[2..4]).toManaged(limbs_buffer.allocator);
                const Cp = fixedIntFromSignedDoubleLimb(C, storage[4..6]).toManaged(limbs_buffer.allocator);
                const Dp = fixedIntFromSignedDoubleLimb(D, storage[6..8]).toManaged(limbs_buffer.allocator);

                // t_big = Ax + By
                try r.mul(&x, &Ap);
                try t_big.mul(&y, &Bp);
                try t_big.add(&r, &t_big);

                // u = Cx + Dy, r as u
                try tmp_x.copy(x.toConst());
                try x.mul(&tmp_x, &Cp);
                try r.mul(&y, &Dp);
                try r.add(&x, &r);

                x.swap(&t_big);
                y.swap(&r);
            }
        }

        // euclidean algorithm
        assert(x.toConst().order(y.toConst()) != .lt);

        while (!y.toConst().eqlZero()) {
            try t_big.divTrunc(&r, &x, &y);
            x.swap(&y);
            y.swap(&r);
        }

        result.copy(x.toConst());
    }

    // Truncates by default.
    fn div(q: *Mutable, r: *Mutable, x: *Mutable, y: *Mutable) void {
        assert(!y.eqlZero()); // division by zero
        assert(q != r); // illegal aliasing

        const q_sign = Sign.fromBool(x.sign() == y.sign());
        const r_sign = Sign.fromBool(x.sign() == .pos);

        if (x.toConst().orderAbs(y.toConst()) == .lt) {
            // q may alias x so handle r first.
            r.copy(x.toConst());
            r.setSign(r_sign);

            q.set(0);
            return;
        }

        // Handle trailing zero-words of divisor/dividend. These are not handled in the following
        // algorithms.
        // Note, there must be a non-zero limb for either.
        // const x_trailing = std.mem.indexOfScalar(Limb, x.limbs[0..x.len], 0).?;
        // const y_trailing = std.mem.indexOfScalar(Limb, y.limbs[0..y.len], 0).?;

        const x_trailing = for (x.limbs[0..x.len()], 0..) |xi, i| {
            if (xi != 0) break i;
        } else unreachable;

        const y_trailing = for (y.limbs[0..y.len()], 0..) |yi, i| {
            if (yi != 0) break i;
        } else unreachable;

        const xy_trailing = @min(x_trailing, y_trailing);

        if (y.len() - xy_trailing == 1) {
            const divisor = y.limbs[y.len() - 1];

            // Optimization for small divisor. By using a half limb we can avoid requiring DoubleLimb
            // divisions in the hot code path. This may often require compiler_rt software-emulation.
            if (divisor < maxInt(HalfLimb)) {
                lldiv0p5(q.limbs, &r.limbs[0], x.limbs[xy_trailing..x.len()], @as(HalfLimb, @intCast(divisor)));
            } else {
                lldiv1(q.limbs, &r.limbs[0], x.limbs[xy_trailing..x.len()], divisor);
            }

            q.normalize(x.len() - xy_trailing);
            q.setSign(q_sign);

            r.metadata = Metadata.init(r_sign, 1);
        } else {
            // Shrink x, y such that the trailing zero limbs shared between are removed.
            var x0 = Mutable{
                .limbs = x.limbs[xy_trailing..],
                .metadata = Metadata.init(.pos, x.len() - xy_trailing),
            };

            var y0 = Mutable{
                .limbs = y.limbs[xy_trailing..],
                .metadata = Metadata.init(.pos, y.len() - xy_trailing),
            };

            divmod(q, r, &x0, &y0);
            q.setSign(q_sign);
            r.setSign(r_sign);
        }

        if (xy_trailing != 0 and r.limbs[r.len() - 1] != 0) {
            // Manually shift here since we know its limb aligned.
            mem.copyBackwards(Limb, r.limbs[xy_trailing..], r.limbs[0..r.len()]);
            @memset(r.limbs[0..xy_trailing], 0);
            r.setLen(r.len() + xy_trailing);
        }
    }

    /// Handbook of Applied Cryptography, 14.20
    ///
    /// x = qy + r where 0 <= r < y
    /// y is modified but returned intact.
    fn divmod(
        q: *Mutable,
        r: *Mutable,
        x: *Mutable,
        y: *Mutable,
    ) void {
        // 0.
        // Normalize so that y[t] > b/2
        const lz = @clz(y.limbs[y.len() - 1]);
        const norm_shift = if (lz == 0 and y.toConst().isOdd())
            limb_bits // Force an extra limb so that y is even.
        else
            lz;

        x.shiftLeft(x.toConst(), norm_shift);
        y.shiftLeft(y.toConst(), norm_shift);

        const n = x.len() - 1;
        const t = y.len() - 1;
        const shift = n - t;

        // 1.
        // for 0 <= j <= n - t, set q[j] to 0
        q.metadata = Metadata.init(.pos, shift + 1);
        @memset(q.limbs[0..q.len()], 0);

        // 2.
        // while x >= y * b^(n - t):
        //    x -= y * b^(n - t)
        //    q[n - t] += 1
        // Note, this algorithm is performed only once if y[t] > base/2 and y is even, which we
        // enforced in step 0. This means we can replace the while with an if.
        // Note, multiplication by b^(n - t) comes down to shifting to the right by n - t limbs.
        // We can also replace x >= y * b^(n - t) by x/b^(n - t) >= y, and use shifts for that.
        {
            // x >= y * b^(n - t) can be replaced by x/b^(n - t) >= y.

            // 'divide' x by b^(n - t)
            var tmp = Mutable{
                .limbs = x.limbs[shift..],
                .metadata = Metadata.init(.pos, x.len() - shift),
            };

            if (tmp.toConst().order(y.toConst()) != .lt) {
                // Perform x -= y * b^(n - t)
                // Note, we can subtract y from x[n - t..] and get the result without shifting.
                // We can also re-use tmp which already contains the relevant part of x. Note that
                // this also edits x.
                // Due to the check above, this cannot underflow.
                tmp.sub(tmp.toConst(), y.toConst());

                // tmp.sub normalized tmp, but we need to normalize x now.
                x.limbs.len = tmp.limbs.len + shift;

                q.limbs[shift] += 1;
            }
        }

        // 3.
        // for i from n down to t + 1, do
        var i = n;
        while (i >= t + 1) : (i -= 1) {
            const k = i - t - 1;
            // 3.1.
            // if x_i == y_t:
            //   q[i - t - 1] = b - 1
            // else:
            //   q[i - t - 1] = (x[i] * b + x[i - 1]) / y[t]
            if (x.limbs[i] == y.limbs[t]) {
                q.limbs[k] = maxInt(Limb);
            } else {
                const q0 = (@as(DoubleLimb, x.limbs[i]) << limb_bits) | @as(DoubleLimb, x.limbs[i - 1]);
                const n0 = @as(DoubleLimb, y.limbs[t]);
                q.limbs[k] = @as(Limb, @intCast(q0 / n0));
            }

            // 3.2
            // while q[i - t - 1] * (y[t] * b + y[t - 1] > x[i] * b * b + x[i - 1] + x[i - 2]:
            //   q[i - t - 1] -= 1
            // Note, if y[t] > b / 2 this part is repeated no more than twice.

            // Extract from y.
            const y0 = if (t > 0) y.limbs[t - 1] else 0;
            const y1 = y.limbs[t];

            // Extract from x.
            // Note, big endian.
            const tmp0 = [_]Limb{
                x.limbs[i],
                if (i >= 1) x.limbs[i - 1] else 0,
                if (i >= 2) x.limbs[i - 2] else 0,
            };

            while (true) {
                // Ad-hoc 2x1 multiplication with q[i - t - 1].
                // Note, big endian.
                var tmp1 = [_]Limb{ 0, undefined, undefined };
                tmp1[2] = addMulLimbWithCarry(0, y0, q.limbs[k], &tmp1[0]);
                tmp1[1] = addMulLimbWithCarry(0, y1, q.limbs[k], &tmp1[0]);

                // Big-endian compare
                if (mem.order(Limb, &tmp1, &tmp0) != .gt)
                    break;

                q.limbs[k] -= 1;
            }

            // 3.3.
            // x -= q[i - t - 1] * y * b^(i - t - 1)
            // Note, we multiply by a single limb here.
            // The shift doesn't need to be performed if we add the result of the first multiplication
            // to x[i - t - 1].
            const underflow = llmulLimb(.sub, x.limbs[k..x.len()], y.limbs[0..y.len()], q.limbs[k]);

            // 3.4.
            // if x < 0:
            //   x += y * b^(i - t - 1)
            //   q[i - t - 1] -= 1
            // Note, we check for x < 0 using the underflow flag from the previous operation.
            if (underflow) {
                // While we didn't properly set the signedness of x, this operation should 'flow' it back to positive.
                llaccum(.add, x.limbs[k..x.len()], y.limbs[0..y.len()]);
                q.limbs[k] -= 1;
            }
        }

        x.normalize(x.len());
        q.normalize(q.len());

        // De-normalize r and y.
        r.shiftRight(x.toConst(), norm_shift);
        y.shiftRight(y.toConst(), norm_shift);
    }

    /// If a is positive, this passes through to truncate.
    /// If a is negative, then r is set to positive with the bit pattern ~(a - 1).
    /// r may alias a.
    ///
    /// Asserts `r` has enough storage to store the result.
    /// The upper bound is `calcTwosCompLimbCount(a.len)`.
    pub fn convertToTwosComplement(r: *Mutable, a: Const, signedness: Signedness, bit_count: usize) void {
        if (a.sign() == .pos) {
            r.truncate(a, signedness, bit_count);
            return;
        }

        const req_limbs = calcTwosCompLimbCount(bit_count);
        if (req_limbs == 0 or a.eqlZero()) {
            r.set(0);
            return;
        }

        const bit = @as(Log2Limb, @truncate(bit_count - 1));
        const signmask = @as(Limb, 1) << bit;
        const mask = (signmask << 1) -% 1;

        r.addScalar(a.abs(), -1);
        if (req_limbs > r.len()) {
            @memset(r.limbs[r.len()..req_limbs], 0);
        }

        assert(r.limbs.len >= req_limbs);
        r.setLen(req_limbs);

        llnot(r.limbs[0..r.len()]);
        r.limbs[r.len() - 1] &= mask;
        r.normalize(r.len());
    }

    /// Truncate an integer to a number of bits, following 2s-complement semantics.
    /// r may alias a.
    ///
    /// Asserts `r` has enough storage to store the result.
    /// The upper bound is `calcTwosCompLimbCount(a.len)`.
    pub fn truncate(r: *Mutable, a: Const, signedness: Signedness, bit_count: usize) void {
        const req_limbs = calcTwosCompLimbCount(bit_count);

        // Handle 0-bit integers.
        if (req_limbs == 0 or a.eqlZero()) {
            r.set(0);
            return;
        }

        const bit = @as(Log2Limb, @truncate(bit_count - 1));
        const signmask = @as(Limb, 1) << bit; // 0b0..010...0 where 1 is the sign bit.
        const mask = (signmask << 1) -% 1; // 0b0..01..1 where the leftmost 1 is the sign bit.

        if (a.sign() == .neg) {
            // Convert the integer from sign-magnitude into twos-complement.
            // -x = ~(x - 1)
            // Note, we simply take req_limbs * @bitSizeOf(Limb) as the
            // target bit count.

            r.addScalar(a.abs(), -1);

            // Zero-extend the result
            if (req_limbs > r.len()) {
                @memset(r.limbs[r.len()..req_limbs], 0);
            }

            // Truncate to required number of limbs.
            assert(r.limbs.len >= req_limbs);
            r.setLen(req_limbs);

            // Without truncating, we can already peek at the sign bit of the result here.
            // Note that it will be 0 if the result is negative, as we did not apply the flip here.
            // If the result is negative, we have
            // -(-x & mask)
            // = ~(~(x - 1) & mask) + 1
            // = ~(~((x - 1) | ~mask)) + 1
            // = ((x - 1) | ~mask)) + 1
            // Note, this is only valid for the target bits and not the upper bits
            // of the most significant limb. Those still need to be cleared.
            // Also note that `mask` is zero for all other bits, reducing to the identity.
            // This means that we still need to use & mask to clear off the upper bits.

            if (signedness == .signed and r.limbs[r.len() - 1] & signmask == 0) {
                // Re-add the one and negate to get the result.
                r.limbs[r.len() - 1] &= mask;
                // Note, addition cannot require extra limbs here as we did a subtraction before.
                r.addScalar(r.toConst(), 1);
                r.normalize(r.len());
                r.setSign(.neg);
            } else {
                llnot(r.limbs[0..r.len()]);
                r.limbs[r.len() - 1] &= mask;
                r.normalize(r.len());
            }
        } else {
            if (a.len() < req_limbs) {
                // Integer fits within target bits, no wrapping required.
                r.copy(a);
                return;
            }

            r.copy(.{
                .limbs = a.limbs,
                .metadata = Metadata.init(.pos, req_limbs),
            });
            r.limbs[r.len() - 1] &= mask;
            r.normalize(r.len());

            if (signedness == .signed and r.limbs[r.len() - 1] & signmask != 0) {
                // Convert 2s-complement back to sign-magnitude.
                // Sign-extend the upper bits so that they are inverted correctly.
                r.limbs[r.len() - 1] |= ~mask;
                llnot(r.limbs[0..r.len()]);

                // Note, can only overflow if r holds 0xFFF...F which can only happen if
                // a holds 0.
                r.addScalar(r.toConst(), 1);
                r.setSign(.neg);
            }
        }
    }

    /// Saturate an integer to a number of bits, following 2s-complement semantics.
    /// r may alias a.
    ///
    /// Asserts `r` has enough storage to store the result.
    /// The upper bound is `calcTwosCompLimbCount(a.len)`.
    pub fn saturate(r: *Mutable, a: Const, signedness: Signedness, bit_count: usize) void {
        if (!a.fitsInTwosComp(signedness, bit_count)) {
            r.setTwosCompIntLimit(if (r.sign() == .pos) .max else .min, signedness, bit_count);
        }
    }

    /// Read the value of `x` from `buffer`.
    /// Asserts that `buffer` is large enough to contain a value of bit-size `bit_count`.
    ///
    /// The contents of `buffer` are interpreted as if they were the contents of
    /// @ptrCast(*[buffer.len]const u8, &x). Byte ordering is determined by `endian`
    /// and any required padding bits are expected on the MSB end.
    pub fn readTwosComplement(
        x: *Mutable,
        buffer: []const u8,
        bit_count: usize,
        endian: Endian,
        signedness: Signedness,
    ) void {
        return readPackedTwosComplement(x, buffer, 0, bit_count, endian, signedness);
    }

    /// Read the value of `x` from a packed memory `buffer`.
    /// Asserts that `buffer` is large enough to contain a value of bit-size `bit_count`
    /// at offset `bit_offset`.
    ///
    /// This is equivalent to loading the value of an integer with `bit_count` bits as
    /// if it were a field in packed memory at the provided bit offset.
    pub fn readPackedTwosComplement(
        x: *Mutable,
        buffer: []const u8,
        bit_offset: usize,
        bit_count: usize,
        endian: Endian,
        signedness: Signedness,
    ) void {
        if (bit_count == 0) {
            x.limbs[0] = 0;
            x.metadata = Metadata.init(.pos, 1);
            return;
        }

        // Check whether the input is negative
        var positive = true;
        if (signedness == .signed) {
            const total_bits = bit_offset + bit_count;
            const last_byte = switch (endian) {
                .little => ((total_bits + 7) / 8) - 1,
                .big => buffer.len - ((total_bits + 7) / 8),
            };

            const sign_bit = @as(u8, 1) << @as(u3, @intCast((total_bits - 1) % 8));
            positive = ((buffer[last_byte] & sign_bit) == 0);
        }

        // Copy all complete limbs
        var carry: u1 = 1;
        var limb_index: usize = 0;
        var bit_index: usize = 0;
        while (limb_index < bit_count / @bitSizeOf(Limb)) : (limb_index += 1) {
            // Read one Limb of bits
            var limb = mem.readPackedInt(Limb, buffer, bit_index + bit_offset, endian);
            bit_index += @bitSizeOf(Limb);

            // 2's complement (bitwise not, then add carry bit)
            if (!positive) {
                const ov = @addWithOverflow(~limb, carry);
                limb = ov[0];
                carry = ov[1];
            }
            x.limbs[limb_index] = limb;
        }

        // Copy the remaining bits
        if (bit_count != bit_index) {
            // Read all remaining bits
            var limb = switch (signedness) {
                .unsigned => mem.readVarPackedInt(Limb, buffer, bit_index + bit_offset, bit_count - bit_index, endian, .unsigned),
                .signed => b: {
                    const SLimb = std.meta.Int(.signed, @bitSizeOf(Limb));
                    const limb = mem.readVarPackedInt(SLimb, buffer, bit_index + bit_offset, bit_count - bit_index, endian, .signed);
                    break :b @as(Limb, @bitCast(limb));
                },
            };

            // 2's complement (bitwise not, then add carry bit)
            if (!positive) {
                const ov = @addWithOverflow(~limb, carry);
                assert(ov[1] == 0);
                limb = ov[0];
            }
            x.limbs[limb_index] = limb;

            limb_index += 1;
        }

        x.metadata = Metadata.init(Sign.fromBool(positive), limb_index);
        x.normalize(x.len());
    }

    /// Normalize a possible sequence of leading zeros.
    ///
    /// [1, 2, 3, 4, 0] -> [1, 2, 3, 4]
    /// [1, 2, 0, 0, 0] -> [1, 2]
    /// [0, 0, 0, 0, 0] -> [0]
    pub fn normalize(r: *Mutable, length: usize) void {
        r.setLen(llnormalize(r.limbs[0..length]));
    }
};

/// A arbitrary-precision big integer, with a fixed set of immutable limbs.
pub const Const = struct {
    /// Raw digits. These are:
    ///
    /// * Little-endian ordered
    /// * limbs.len >= 1
    /// * Zero is represented as limbs.len == 1 with limbs[0] == 0.
    ///
    /// Accessing limbs directly should be avoided.
    limbs: [*]const Limb,
    metadata: Metadata,

    /// Returns the number of limbs used to represent the current value.
    pub fn len(self: Const) usize {
        return self.metadata.len();
    }

    /// Returns the sign of the current value.
    pub fn sign(self: Const) Sign {
        return self.metadata.sign();
    }

    /// The result is an independent resource which is managed by the caller.
    pub fn toManaged(self: Const, allocator: Allocator) Allocator.Error!Managed {
        const limbs = try allocator.alloc(Limb, @max(Managed.default_capacity, self.len()));
        @memcpy(limbs[0..self.len()], self.limbs);
        return Managed{
            .allocator = allocator,
            .limbs = limbs,
            .metadata = self.metadata,
        };
    }

    /// Asserts `limbs` is big enough to store the value.
    pub fn toMutable(self: Const, limbs: []Limb) Mutable {
        @memcpy(limbs[0..self.len()], self.limbs[0..self.len()]);
        return .{
            .limbs = limbs,
            .metadata = self.metadata,
        };
    }

    pub fn dump(self: Const) void {
        for (self.limbs[0..self.len()]) |limb| {
            std.debug.print("{x} ", .{limb});
        }
        std.debug.print("positive={}\n", .{self.sign() == .pos});
    }

    pub fn abs(self: Const) Const {
        return .{
            .limbs = self.limbs,
            .metadata = self.metadata.withSign(.pos),
        };
    }

    pub fn negate(self: Const) Const {
        return .{
            .limbs = self.limbs,
            .metadata = self.metadata.withSign(self.sign().negate()),
        };
    }

    pub fn isOdd(self: Const) bool {
        return self.limbs[0] & 1 != 0;
    }

    pub fn isEven(self: Const) bool {
        return !self.isOdd();
    }

    /// Returns the number of bits required to represent the absolute value of an integer.
    pub fn bitCountAbs(self: Const) usize {
        return (self.len() - 1) * limb_bits + (limb_bits - @clz(self.limbs[self.len() - 1]));
    }

    /// Returns the number of bits required to represent the integer in twos-complement form.
    ///
    /// If the integer is negative the value returned is the number of bits needed by a signed
    /// integer to represent the value. If positive the value is the number of bits for an
    /// unsigned integer. Any unsigned integer will fit in the signed integer with bitcount
    /// one greater than the returned value.
    ///
    /// e.g. -127 returns 8 as it will fit in an i8. 127 returns 7 since it fits in a u7.
    pub fn bitCountTwosComp(self: Const) usize {
        var bits = self.bitCountAbs();

        // If the entire value has only one bit set (e.g. 0b100000000) then the negation in twos
        // complement requires one less bit.
        if (self.sign() == .neg) block: {
            bits += 1;

            if (@popCount(self.limbs[self.len() - 1]) == 1) {
                for (self.limbs[0 .. self.len() - 1]) |limb| {
                    if (@popCount(limb) != 0) {
                        break :block;
                    }
                }

                bits -= 1;
            }
        }

        return bits;
    }

    /// @popCount with two's complement semantics.
    ///
    /// This returns the number of 1 bits set when the value would be represented in
    /// two's complement with the given integer width (bit_count).
    /// This includes the leading sign bit, which will be set for negative values.
    ///
    /// Asserts that bit_count is enough to represent value in two's compliment
    /// and that the final result fits in a usize.
    /// Asserts that there are no trailing empty limbs on the most significant end,
    /// i.e. that limb count matches `calcLimbLen()` and zero is not negative.
    pub fn popCount(self: Const, bit_count: usize) usize {
        var sum: usize = 0;
        if (self.sign() == .pos) {
            for (self.limbs[0..self.len()]) |limb| {
                sum += @popCount(limb);
            }
        } else {
            assert(self.fitsInTwosComp(.signed, bit_count));
            assert(self.limbs[self.len() - 1] != 0);

            var remaining_bits = bit_count;
            var carry: u1 = 1;
            var add_res: Limb = undefined;

            // All but the most significant limb.
            for (self.limbs[0 .. self.len() - 1]) |limb| {
                const ov = @addWithOverflow(~limb, carry);
                add_res = ov[0];
                carry = ov[1];
                sum += @popCount(add_res);
                remaining_bits -= limb_bits; // Asserted not to underflow by fitsInTwosComp
            }

            // The most significant limb may have fewer than @bitSizeOf(Limb) meaningful bits,
            // which we can detect with @clz().
            // There may also be fewer limbs than needed to fill bit_count.
            const limb = self.limbs[self.len() - 1];
            const leading_zeroes = @clz(limb);
            // The most significant limb is asserted not to be all 0s (above),
            // so ~limb cannot be all 1s, and ~limb + 1 cannot overflow.
            sum += @popCount(~limb + carry);
            sum -= leading_zeroes; // All leading zeroes were flipped and added to sum, so undo those
            const remaining_ones = remaining_bits - (limb_bits - leading_zeroes); // All bits not covered by limbs
            sum += remaining_ones;
        }
        return sum;
    }

    pub fn fitsInTwosComp(self: Const, signedness: Signedness, bit_count: usize) bool {
        if (self.eqlZero()) {
            return true;
        }
        if (signedness == .unsigned and self.sign() == .neg) {
            return false;
        }

        const req_bits = self.bitCountTwosComp() + @intFromBool(self.sign() == .pos and signedness == .signed);
        return bit_count >= req_bits;
    }

    /// Returns whether self can fit into an integer of the requested type.
    pub fn fits(self: Const, comptime T: type) bool {
        const info = @typeInfo(T).Int;
        return self.fitsInTwosComp(info.signedness, info.bits);
    }

    /// Returns the approximate size of the integer in the given base. Negative values accommodate for
    /// the minus sign. This is used for determining the number of characters needed to print the
    /// value. It is inexact and may exceed the given value by ~1-2 bytes.
    /// TODO See if we can make this exact.
    pub fn sizeInBaseUpperBound(self: Const, base: usize) usize {
        const bit_count = @as(usize, @intFromBool(self.sign() == .neg)) + self.bitCountAbs();
        return (bit_count / math.log2(base)) + 2;
    }

    pub const ConvertError = error{
        NegativeIntoUnsigned,
        TargetTooSmall,
    };

    /// Convert self to type T.
    ///
    /// Returns an error if self cannot be narrowed into the requested type without truncation.
    pub fn to(self: Const, comptime T: type) ConvertError!T {
        switch (@typeInfo(T)) {
            .Int => |info| {
                // Make sure -0 is handled correctly.
                if (self.eqlZero()) return 0;

                const UT = std.meta.Int(.unsigned, info.bits);

                if (!self.fitsInTwosComp(info.signedness, info.bits)) {
                    return error.TargetTooSmall;
                }

                var r: UT = 0;

                if (@sizeOf(UT) <= @sizeOf(Limb)) {
                    r = @as(UT, @intCast(self.limbs[0]));
                } else {
                    for (self.limbs[0..self.len()], 0..) |_, ri| {
                        const limb = self.limbs[self.len() - ri - 1];
                        r <<= limb_bits;
                        r |= limb;
                    }
                }

                if (info.signedness == .unsigned) {
                    return if (self.sign() == .pos) @as(T, @intCast(r)) else error.NegativeIntoUnsigned;
                } else {
                    if (self.sign() == .pos) {
                        return @as(T, @intCast(r));
                    } else {
                        if (math.cast(T, r)) |ok| {
                            return -ok;
                        } else {
                            return minInt(T);
                        }
                    }
                }
            },
            else => @compileError("cannot convert Const to type " ++ @typeName(T)),
        }
    }

    /// To allow `std.fmt.format` to work with this type.
    /// If the integer is larger than `pow(2, 64 * @sizeOf(usize) * 8), this function will fail
    /// to print the string, printing "(BigInt)" instead of a number.
    /// This is because the rendering algorithm requires reversing a string, which requires O(N) memory.
    /// See `toString` and `toStringAlloc` for a way to print big integers without failure.
    pub fn format(
        self: Const,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        out_stream: anytype,
    ) !void {
        _ = options;
        comptime var base = 10;
        comptime var case: std.fmt.Case = .lower;

        if (fmt.len == 0 or comptime mem.eql(u8, fmt, "d")) {
            base = 10;
            case = .lower;
        } else if (comptime mem.eql(u8, fmt, "b")) {
            base = 2;
            case = .lower;
        } else if (comptime mem.eql(u8, fmt, "x")) {
            base = 16;
            case = .lower;
        } else if (comptime mem.eql(u8, fmt, "X")) {
            base = 16;
            case = .upper;
        } else {
            std.fmt.invalidFmtError(fmt, self);
        }

        var limbs: [128]Limb = undefined;
        const needed_limbs = calcDivLimbsBufferLen(self.len(), 1);
        if (needed_limbs > limbs.len)
            return out_stream.writeAll("(BigInt)");

        // This is the inverse of calcDivLimbsBufferLen
        const available_len = (limbs.len / 3) - 2;

        const biggest: Const = .{
            .limbs = &([1]Limb{comptime math.maxInt(Limb)} ** available_len),
            .metadata = Metadata.init(.neg, available_len),
        };
        var buf: [biggest.sizeInBaseUpperBound(base)]u8 = undefined;
        const out_len = self.toString(&buf, base, case, &limbs);
        return out_stream.writeAll(buf[0..out_len]);
    }

    /// Converts self to a string in the requested base.
    /// Caller owns returned memory.
    /// Asserts that `base` is in the range [2, 16].
    /// See also `toString`, a lower level function than this.
    pub fn toStringAlloc(self: Const, allocator: Allocator, base: u8, case: std.fmt.Case) Allocator.Error![]u8 {
        assert(base >= 2);
        assert(base <= 16);

        if (self.eqlZero()) {
            return allocator.dupe(u8, "0");
        }
        const string = try allocator.alloc(u8, self.sizeInBaseUpperBound(base));
        errdefer allocator.free(string);

        const limbs = try allocator.alloc(Limb, calcToStringLimbsBufferLen(self.len(), base));
        defer allocator.free(limbs);

        return allocator.realloc(string, self.toString(string, base, case, limbs));
    }

    /// Converts self to a string in the requested base.
    /// Asserts that `base` is in the range [2, 16].
    /// `string` is a caller-provided slice of at least `sizeInBaseUpperBound` bytes,
    /// where the result is written to.
    /// Returns the length of the string.
    /// `limbs_buffer` is caller-provided memory for `toString` to use as a working area. It must have
    /// length of at least `calcToStringLimbsBufferLen`.
    /// In the case of power-of-two base, `limbs_buffer` is ignored.
    /// See also `toStringAlloc`, a higher level function than this.
    pub fn toString(self: Const, string: []u8, base: u8, case: std.fmt.Case, limbs_buffer: []Limb) usize {
        assert(base >= 2);
        assert(base <= 16);

        if (self.eqlZero()) {
            string[0] = '0';
            return 1;
        }

        var digits_len: usize = 0;

        // Power of two: can do a single pass and use masks to extract digits.
        if (math.isPowerOfTwo(base)) {
            const base_shift = math.log2_int(Limb, base);

            outer: for (self.limbs[0..self.len()]) |limb| {
                var shift: usize = 0;
                while (shift < limb_bits) : (shift += base_shift) {
                    const r = @as(u8, @intCast((limb >> @as(Log2Limb, @intCast(shift))) & @as(Limb, base - 1)));
                    const ch = std.fmt.digitToChar(r, case);
                    string[digits_len] = ch;
                    digits_len += 1;
                    // If we hit the end, it must be all zeroes from here.
                    if (digits_len == string.len) break :outer;
                }
            }

            // Always will have a non-zero digit somewhere.
            while (string[digits_len - 1] == '0') {
                digits_len -= 1;
            }
        } else {
            // Non power-of-two: batch divisions per word size.
            // We use a HalfLimb here so the division uses the faster lldiv0p5 over lldiv1 codepath.
            const digits_per_limb = math.log(HalfLimb, base, maxInt(HalfLimb));
            var limb_base: Limb = 1;
            var j: usize = 0;
            while (j < digits_per_limb) : (j += 1) {
                limb_base *= base;
            }
            const b: Const = .{ .limbs = &[_]Limb{limb_base}, .metadata = Metadata.init(.pos, 1) };

            var q: Mutable = .{
                .limbs = limbs_buffer[0 .. self.len() + 2],
                .metadata = Metadata.init(.pos, self.len()), // Make absolute by ignoring self.positive.
            };
            @memcpy(q.limbs[0..self.len()], self.limbs);

            var r: Mutable = .{
                .limbs = limbs_buffer[q.len()..][0..self.len()],
                .metadata = Metadata.init(.pos, 1),
            };
            r.limbs[0] = 0;

            const rest_of_the_limbs_buf = limbs_buffer[q.len() + r.len() ..];

            while (q.len() >= 2) {
                // Passing an allocator here would not be helpful since this division is destroying
                // information, not creating it. [TODO citation needed]
                q.divTrunc(&r, q.toConst(), b, rest_of_the_limbs_buf);

                var r_word = r.limbs[0];
                var i: usize = 0;
                while (i < digits_per_limb) : (i += 1) {
                    const ch = std.fmt.digitToChar(@as(u8, @intCast(r_word % base)), case);
                    r_word /= base;
                    string[digits_len] = ch;
                    digits_len += 1;
                }
            }

            {
                assert(q.len() == 1);

                var r_word = q.limbs[0];
                while (r_word != 0) {
                    const ch = std.fmt.digitToChar(@as(u8, @intCast(r_word % base)), case);
                    r_word /= base;
                    string[digits_len] = ch;
                    digits_len += 1;
                }
            }
        }

        if (self.sign() == .neg) {
            string[digits_len] = '-';
            digits_len += 1;
        }

        const s = string[0..digits_len];
        mem.reverse(u8, s);
        return s.len;
    }

    /// Write the value of `x` into `buffer`
    /// Asserts that `buffer` is large enough to store the value.
    ///
    /// `buffer` is filled so that its contents match what would be observed via
    /// @ptrCast(*[buffer.len]const u8, &x). Byte ordering is determined by `endian`,
    /// and any required padding bits are added on the MSB end.
    pub fn writeTwosComplement(x: Const, buffer: []u8, endian: Endian) void {
        return writePackedTwosComplement(x, buffer, 0, 8 * buffer.len, endian);
    }

    /// Write the value of `x` to a packed memory `buffer`.
    /// Asserts that `buffer` is large enough to contain a value of bit-size `bit_count`
    /// at offset `bit_offset`.
    ///
    /// This is equivalent to storing the value of an integer with `bit_count` bits as
    /// if it were a field in packed memory at the provided bit offset.
    pub fn writePackedTwosComplement(x: Const, buffer: []u8, bit_offset: usize, bit_count: usize, endian: Endian) void {
        assert(x.fitsInTwosComp(if (x.sign() == .pos) .unsigned else .signed, bit_count));

        // Copy all complete limbs
        var carry: u1 = 1;
        var limb_index: usize = 0;
        var bit_index: usize = 0;
        while (limb_index < bit_count / @bitSizeOf(Limb)) : (limb_index += 1) {
            var limb: Limb = if (limb_index < x.len()) x.limbs[limb_index] else 0;

            // 2's complement (bitwise not, then add carry bit)
            if (x.sign() == .neg) {
                const ov = @addWithOverflow(~limb, carry);
                limb = ov[0];
                carry = ov[1];
            }

            // Write one Limb of bits
            mem.writePackedInt(Limb, buffer, bit_index + bit_offset, limb, endian);
            bit_index += @bitSizeOf(Limb);
        }

        // Copy the remaining bits
        if (bit_count != bit_index) {
            var limb: Limb = if (limb_index < x.len()) x.limbs[limb_index] else 0;

            // 2's complement (bitwise not, then add carry bit)
            if (x.sign() == .neg) limb = ~limb +% carry;

            // Write all remaining bits
            mem.writeVarPackedInt(buffer, bit_index + bit_offset, bit_count - bit_index, limb, endian);
        }
    }

    /// Returns `math.Order.lt`, `math.Order.eq`, `math.Order.gt` if
    /// `|a| < |b|`, `|a| == |b|`, or `|a| > |b|` respectively.
    pub fn orderAbs(a: Const, b: Const) math.Order {
        if (a.len() < b.len()) {
            return .lt;
        }
        if (a.len() > b.len()) {
            return .gt;
        }

        var i: usize = a.len() - 1;
        while (i != 0) : (i -= 1) {
            if (a.limbs[i] != b.limbs[i]) {
                break;
            }
        }

        if (a.limbs[i] < b.limbs[i]) {
            return .lt;
        } else if (a.limbs[i] > b.limbs[i]) {
            return .gt;
        } else {
            return .eq;
        }
    }

    /// Returns `math.Order.lt`, `math.Order.eq`, `math.Order.gt` if `a < b`, `a == b` or `a > b` respectively.
    pub fn order(a: Const, b: Const) math.Order {
        if (a.sign() != b.sign()) {
            if (eqlZero(a) and eqlZero(b)) {
                return .eq;
            } else {
                return if (a.sign() == .pos) .gt else .lt;
            }
        } else {
            const r = orderAbs(a, b);
            return if (a.sign() == .pos) r else switch (r) {
                .lt => math.Order.gt,
                .eq => math.Order.eq,
                .gt => math.Order.lt,
            };
        }
    }

    /// Same as `order` but the right-hand operand is a primitive integer.
    pub fn orderAgainstScalar(lhs: Const, scalar: anytype) math.Order {
        // Normally we could just determine the number of limbs needed with calcLimbLen,
        // but that is not comptime-known when scalar is not a comptime_int.  Instead, we
        // use calcTwosCompLimbCount for a non-comptime_int scalar, which can be pessimistic
        // in the case that scalar happens to be small in magnitude within its type, but it
        // is well worth being able to use the stack and not needing an allocator passed in.
        // Note that Mutable.init still sets len to calcLimbLen(scalar) in any case.
        const limb_len = comptime switch (@typeInfo(@TypeOf(scalar))) {
            .ComptimeInt => calcLimbLen(scalar),
            .Int => |info| calcTwosCompLimbCount(info.bits),
            else => @compileError("expected scalar to be an int"),
        };
        var limbs: [limb_len]Limb = undefined;
        const rhs = Mutable.init(&limbs, scalar);
        return order(lhs, rhs.toConst());
    }

    /// Returns true if `a == 0`.
    pub fn eqlZero(a: Const) bool {
        var d: Limb = 0;
        for (a.limbs[0..a.len()]) |limb| d |= limb;
        return d == 0;
    }

    /// Returns true if `|a| == |b|`.
    pub fn eqlAbs(a: Const, b: Const) bool {
        return orderAbs(a, b) == .eq;
    }

    /// Returns true if `a == b`.
    pub fn eql(a: Const, b: Const) bool {
        return order(a, b) == .eq;
    }

    pub fn clz(a: Const, bits: Limb) Limb {
        // Limbs are stored in little-endian order but we need
        // to iterate big-endian.
        var total_limb_lz: Limb = 0;
        var i: usize = a.len();
        const bits_per_limb = @sizeOf(Limb) * 8;
        while (i != 0) {
            i -= 1;
            const limb = a.limbs[i];
            const this_limb_lz = @clz(limb);
            total_limb_lz += this_limb_lz;
            if (this_limb_lz != bits_per_limb) break;
        }
        const total_limb_bits = a.len() * bits_per_limb;
        return total_limb_lz + bits - total_limb_bits;
    }

    pub fn ctz(a: Const, bits: Limb) Limb {
        // Limbs are stored in little-endian order.
        var result: Limb = 0;
        for (a.limbs) |limb| {
            const limb_tz = @ctz(limb);
            result += limb_tz;
            if (limb_tz != @sizeOf(Limb) * 8) break;
        }
        return @min(result, bits);
    }
};

/// An arbitrary-precision big integer along with an allocator which manages the memory.
///
/// Memory is allocated as needed to ensure operations never overflow. The range
/// is bounded only by available memory.
pub const Managed = struct {
    pub const sign_bit: usize = 1 << (@typeInfo(usize).Int.bits - 1);

    /// Default number of limbs to allocate on creation of a `Managed`.
    pub const default_capacity = 4;

    /// Allocator used by the Managed when requesting memory.
    allocator: Allocator,

    /// Raw digits. These are:
    ///
    /// * Little-endian ordered
    /// * limbs.len >= 1
    /// * Zero is represent as Managed.len() == 1 with limbs[0] == 0.
    ///
    /// Accessing limbs directly should be avoided.
    limbs: []Limb,
    metadata: Metadata,

    /// Creates a new `Managed`. `default_capacity` limbs will be allocated immediately.
    /// The integer value after initializing is `0`.
    pub fn init(allocator: Allocator) !Managed {
        return initCapacity(allocator, default_capacity);
    }

    pub fn toMutable(self: Managed) Mutable {
        return .{
            .limbs = self.limbs,
            .metadata = self.metadata,
        };
    }

    pub fn toConst(self: Managed) Const {
        return .{
            .limbs = self.limbs.ptr,
            .metadata = self.metadata,
        };
    }

    /// Creates a new `Managed` with value `value`.
    ///
    /// This is identical to an `init`, followed by a `set`.
    pub fn initSet(allocator: Allocator, value: anytype) !Managed {
        var s = try Managed.init(allocator);
        errdefer s.deinit();
        try s.set(value);
        return s;
    }

    /// Creates a new Managed with a specific capacity. If capacity < default_capacity then the
    /// default capacity will be used instead.
    /// The integer value after initializing is `0`.
    pub fn initCapacity(allocator: Allocator, capacity: usize) !Managed {
        return Managed{
            .allocator = allocator,
            .metadata = Metadata.init(.pos, 1),
            .limbs = block: {
                const limbs = try allocator.alloc(Limb, @max(default_capacity, capacity));
                limbs[0] = 0;
                break :block limbs;
            },
        };
    }

    /// Returns the number of limbs currently in use.
    pub fn len(self: Managed) usize {
        return self.metadata.len();
    }

    /// Returns the sign of self
    pub fn sign(self: Managed) Sign {
        return self.metadata.sign();
    }

    /// Sets the length of an Managed.
    ///
    /// If setLen is used, then the Managed must be normalized to suit.
    pub fn setLen(self: *Managed, new_len: usize) void {
        self.metadata = self.metadata.withLen(new_len);
    }

    /// Sets the sign of an Managed.
    pub fn setSign(self: *Managed, new_sign: Sign) void {
        self.metadata = self.metadata.withSign(new_sign);
    }

    /// Ensures an Managed has enough space allocated for capacity limbs. If the Managed does not have
    /// sufficient capacity, the exact amount will be allocated. This occurs even if the requested
    /// capacity is only greater than the current capacity by one limb.
    pub fn ensureCapacity(self: *Managed, capacity: usize) !void {
        if (capacity <= self.limbs.len) {
            return;
        }
        self.limbs = try self.allocator.realloc(self.limbs, capacity);
    }

    /// Frees all associated memory.
    pub fn deinit(self: *Managed) void {
        self.allocator.free(self.limbs);
        self.* = undefined;
    }

    /// Returns a `Managed` with the same value. The returned `Managed` is a deep copy and
    /// can be modified separately from the original, and its resources are managed
    /// separately from the original.
    pub fn clone(other: Managed) !Managed {
        return other.cloneWithDifferentAllocator(other.allocator);
    }

    pub fn cloneWithDifferentAllocator(other: Managed, allocator: Allocator) !Managed {
        return Managed{
            .allocator = allocator,
            .metadata = other.metadata,
            .limbs = block: {
                const limbs = try allocator.alloc(Limb, other.len());
                @memcpy(limbs, other.limbs[0..other.len()]);
                break :block limbs;
            },
        };
    }

    /// Copies the value of the integer to an existing `Managed` so that they both have the same value.
    /// Extra memory will be allocated if the receiver does not have enough capacity.
    pub fn copy(self: *Managed, other: Const) !void {
        if (self.limbs.ptr == other.limbs) return;

        try self.ensureCapacity(other.len());
        @memcpy(self.limbs[0..other.len()], other.limbs[0..other.len()]);
        self.metadata = other.metadata;
    }

    /// Efficiently swap a `Managed` with another. This swaps the limb pointers and a full copy is not
    /// performed. The address of the limbs field will not be the same after this function.
    pub fn swap(self: *Managed, other: *Managed) void {
        mem.swap(Managed, self, other);
    }

    /// Debugging tool: prints the state to stderr.
    pub fn dump(self: Managed) void {
        for (self.limbs[0..self.len()]) |limb| {
            std.debug.print("{x} ", .{limb});
        }
        std.debug.print("capacity={} positive={}\n", .{ self.limbs.len, self.sign() == .pos });
    }

    /// Negate the sign.
    pub fn negate(self: *Managed) void {
        self.setSign(self.sign().negate());
    }

    /// Make positive.
    pub fn abs(self: *Managed) void {
        self.setSign(.pos);
    }

    pub fn isOdd(self: Managed) bool {
        return self.limbs[0] & 1 != 0;
    }

    pub fn isEven(self: Managed) bool {
        return !self.isOdd();
    }

    /// Returns the number of bits required to represent the absolute value of an integer.
    pub fn bitCountAbs(self: Managed) usize {
        return self.toConst().bitCountAbs();
    }

    /// Returns the number of bits required to represent the integer in twos-complement form.
    ///
    /// If the integer is negative the value returned is the number of bits needed by a signed
    /// integer to represent the value. If positive the value is the number of bits for an
    /// unsigned integer. Any unsigned integer will fit in the signed integer with bitcount
    /// one greater than the returned value.
    ///
    /// e.g. -127 returns 8 as it will fit in an i8. 127 returns 7 since it fits in a u7.
    pub fn bitCountTwosComp(self: Managed) usize {
        return self.toConst().bitCountTwosComp();
    }

    pub fn fitsInTwosComp(self: Managed, signedness: Signedness, bit_count: usize) bool {
        return self.toConst().fitsInTwosComp(signedness, bit_count);
    }

    /// Returns whether self can fit into an integer of the requested type.
    pub fn fits(self: Managed, comptime T: type) bool {
        return self.toConst().fits(T);
    }

    /// Returns the approximate size of the integer in the given base. Negative values accommodate for
    /// the minus sign. This is used for determining the number of characters needed to print the
    /// value. It is inexact and may exceed the given value by ~1-2 bytes.
    pub fn sizeInBaseUpperBound(self: Managed, base: usize) usize {
        return self.toConst().sizeInBaseUpperBound(base);
    }

    /// Sets an Managed to value. Value must be an primitive integer type.
    pub fn set(self: *Managed, value: anytype) Allocator.Error!void {
        try self.ensureCapacity(calcLimbLen(value));
        var m = self.toMutable();
        m.set(value);
        self.metadata = m.metadata;
    }

    pub const ConvertError = Const.ConvertError;

    /// Convert self to type T.
    ///
    /// Returns an error if self cannot be narrowed into the requested type without truncation.
    pub fn to(self: Managed, comptime T: type) ConvertError!T {
        return self.toConst().to(T);
    }

    /// Set self from the string representation `value`.
    ///
    /// `value` must contain only digits <= `base` and is case insensitive.  Base prefixes are
    /// not allowed (e.g. 0x43 should simply be 43).  Underscores in the input string are
    /// ignored and can be used as digit separators.
    ///
    /// Returns an error if memory could not be allocated or `value` has invalid digits for the
    /// requested base.
    ///
    /// self's allocator is used for temporary storage to boost multiplication performance.
    pub fn setString(self: *Managed, base: u8, value: []const u8) !void {
        if (base < 2 or base > 16) return error.InvalidBase;
        try self.ensureCapacity(calcSetStringLimbCount(base, value.len));
        const limbs_buffer = try self.allocator.alloc(Limb, calcSetStringLimbsBufferLen(base, value.len));
        defer self.allocator.free(limbs_buffer);
        var m = self.toMutable();
        try m.setString(base, value, limbs_buffer, self.allocator);
        self.metadata = m.metadata;
    }

    /// Set self to either bound of a 2s-complement integer.
    /// Note: The result is still sign-magnitude, not twos complement! In order to convert the
    /// result to twos complement, it is sufficient to take the absolute value.
    pub fn setTwosCompIntLimit(
        r: *Managed,
        limit: TwosCompIntLimit,
        signedness: Signedness,
        bit_count: usize,
    ) !void {
        try r.ensureCapacity(calcTwosCompLimbCount(bit_count));
        var m = r.toMutable();
        m.setTwosCompIntLimit(limit, signedness, bit_count);
        r.metadata = m.metadata;
    }

    /// Converts self to a string in the requested base. Memory is allocated from the provided
    /// allocator and not the one present in self.
    pub fn toString(self: Managed, allocator: Allocator, base: u8, case: std.fmt.Case) ![]u8 {
        if (base < 2 or base > 16) return error.InvalidBase;
        return self.toConst().toStringAlloc(allocator, base, case);
    }

    /// To allow `std.fmt.format` to work with `Managed`.
    /// If the integer is larger than `pow(2, 64 * @sizeOf(usize) * 8), this function will fail
    /// to print the string, printing "(BigInt)" instead of a number.
    /// This is because the rendering algorithm requires reversing a string, which requires O(N) memory.
    /// See `toString` and `toStringAlloc` for a way to print big integers without failure.
    pub fn format(
        self: Managed,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        out_stream: anytype,
    ) !void {
        return self.toConst().format(fmt, options, out_stream);
    }

    /// Returns math.Order.lt, math.Order.eq, math.Order.gt if |a| < |b|, |a| ==
    /// |b| or |a| > |b| respectively.
    pub fn orderAbs(a: Managed, b: Managed) math.Order {
        return a.toConst().orderAbs(b.toConst());
    }

    /// Returns math.Order.lt, math.Order.eq, math.Order.gt if a < b, a == b or a
    /// > b respectively.
    pub fn order(a: Managed, b: Managed) math.Order {
        return a.toConst().order(b.toConst());
    }

    /// Returns true if a == 0.
    pub fn eqlZero(a: Managed) bool {
        return a.toConst().eqlZero();
    }

    /// Returns true if |a| == |b|.
    pub fn eqlAbs(a: Managed, b: Managed) bool {
        return a.toConst().eqlAbs(b.toConst());
    }

    /// Returns true if a == b.
    pub fn eql(a: Managed, b: Managed) bool {
        return a.toConst().eql(b.toConst());
    }

    /// Normalize a possible sequence of leading zeros.
    ///
    /// [1, 2, 3, 4, 0] -> [1, 2, 3, 4]
    /// [1, 2, 0, 0, 0] -> [1, 2]
    /// [0, 0, 0, 0, 0] -> [0]
    pub fn normalize(r: *Managed, length: usize) void {
        assert(length > 0);
        assert(length <= r.limbs.len);

        var j = length;
        while (j > 0) : (j -= 1) {
            if (r.limbs[j - 1] != 0) {
                break;
            }
        }

        // Handle zero
        r.setLen(if (j != 0) j else 1);
    }

    /// r = a + scalar
    ///
    /// r and a may be aliases.
    ///
    /// Returns an error if memory could not be allocated.
    pub fn addScalar(r: *Managed, a: *const Managed, scalar: anytype) Allocator.Error!void {
        try r.ensureAddScalarCapacity(a.toConst(), scalar);
        var m = r.toMutable();
        m.addScalar(a.toConst(), scalar);
        r.metadata = m.metadata;
    }

    /// r = a + b
    ///
    /// r, a and b may be aliases.
    ///
    /// Returns an error if memory could not be allocated.
    pub fn add(r: *Managed, a: *const Managed, b: *const Managed) Allocator.Error!void {
        try r.ensureAddCapacity(a.toConst(), b.toConst());
        var m = r.toMutable();
        m.add(a.toConst(), b.toConst());
        r.metadata = m.metadata;
    }

    /// r = a + b with 2s-complement wrapping semantics. Returns whether any overflow occurred.
    ///
    /// r, a and b may be aliases.
    ///
    /// Returns an error if memory could not be allocated.
    pub fn addWrap(
        r: *Managed,
        a: *const Managed,
        b: *const Managed,
        signedness: Signedness,
        bit_count: usize,
    ) Allocator.Error!bool {
        try r.ensureTwosCompCapacity(bit_count);
        var m = r.toMutable();
        const wrapped = m.addWrap(a.toConst(), b.toConst(), signedness, bit_count);
        r.metadata = m.metadata;
        return wrapped;
    }

    /// r = a + b with 2s-complement saturating semantics.
    ///
    /// r, a and b may be aliases.
    ///
    /// Returns an error if memory could not be allocated.
    pub fn addSat(r: *Managed, a: *const Managed, b: *const Managed, signedness: Signedness, bit_count: usize) Allocator.Error!void {
        try r.ensureTwosCompCapacity(bit_count);
        var m = r.toMutable();
        m.addSat(a.toConst(), b.toConst(), signedness, bit_count);
        r.metadata = m.metadata;
    }

    /// r = a - b
    ///
    /// r, a and b may be aliases.
    ///
    /// Returns an error if memory could not be allocated.
    pub fn sub(r: *Managed, a: *const Managed, b: *const Managed) !void {
        try r.ensureCapacity(@max(a.len(), b.len()) + 1);
        var m = r.toMutable();
        m.sub(a.toConst(), b.toConst());
        r.metadata = m.metadata;
    }

    /// r = a - b with 2s-complement wrapping semantics. Returns whether any overflow occurred.
    ///
    /// r, a and b may be aliases.
    ///
    /// Returns an error if memory could not be allocated.
    pub fn subWrap(
        r: *Managed,
        a: *const Managed,
        b: *const Managed,
        signedness: Signedness,
        bit_count: usize,
    ) Allocator.Error!bool {
        try r.ensureTwosCompCapacity(bit_count);
        var m = r.toMutable();
        const wrapped = m.subWrap(a.toConst(), b.toConst(), signedness, bit_count);
        r.metadata = m.metadata;
        return wrapped;
    }

    /// r = a - b with 2s-complement saturating semantics.
    ///
    /// r, a and b may be aliases.
    ///
    /// Returns an error if memory could not be allocated.
    pub fn subSat(
        r: *Managed,
        a: *const Managed,
        b: *const Managed,
        signedness: Signedness,
        bit_count: usize,
    ) Allocator.Error!void {
        try r.ensureTwosCompCapacity(bit_count);
        var m = r.toMutable();
        m.subSat(a.toConst(), b.toConst(), signedness, bit_count);
        r.metadata = m.metadata;
    }

    /// rma = a * b
    ///
    /// rma, a and b may be aliases. However, it is more efficient if rma does not alias a or b.
    ///
    /// Returns an error if memory could not be allocated.
    ///
    /// rma's allocator is used for temporary storage to speed up the multiplication.
    pub fn mul(rma: *Managed, a: *const Managed, b: *const Managed) !void {
        var alias_count: usize = 0;
        if (rma.limbs.ptr == a.limbs.ptr)
            alias_count += 1;
        if (rma.limbs.ptr == b.limbs.ptr)
            alias_count += 1;
        try rma.ensureMulCapacity(a.toConst(), b.toConst());
        var m = rma.toMutable();
        if (alias_count == 0) {
            m.mulNoAlias(a.toConst(), b.toConst(), rma.allocator);
        } else {
            const limb_count = calcMulLimbsBufferLen(a.len(), b.len(), alias_count);
            const limbs_buffer = try rma.allocator.alloc(Limb, limb_count);
            defer rma.allocator.free(limbs_buffer);
            m.mul(a.toConst(), b.toConst(), limbs_buffer, rma.allocator);
        }
        rma.metadata = m.metadata;
    }

    /// rma = a * b with 2s-complement wrapping semantics.
    ///
    /// rma, a and b may be aliases. However, it is more efficient if rma does not alias a or b.
    ///
    /// Returns an error if memory could not be allocated.
    ///
    /// rma's allocator is used for temporary storage to speed up the multiplication.
    pub fn mulWrap(
        rma: *Managed,
        a: *const Managed,
        b: *const Managed,
        signedness: Signedness,
        bit_count: usize,
    ) !void {
        var alias_count: usize = 0;
        if (rma.limbs.ptr == a.limbs.ptr)
            alias_count += 1;
        if (rma.limbs.ptr == b.limbs.ptr)
            alias_count += 1;

        try rma.ensureTwosCompCapacity(bit_count);
        var m = rma.toMutable();
        if (alias_count == 0) {
            m.mulWrapNoAlias(a.toConst(), b.toConst(), signedness, bit_count, rma.allocator);
        } else {
            const limb_count = calcMulWrapLimbsBufferLen(bit_count, a.len(), b.len(), alias_count);
            const limbs_buffer = try rma.allocator.alloc(Limb, limb_count);
            defer rma.allocator.free(limbs_buffer);
            m.mulWrap(a.toConst(), b.toConst(), signedness, bit_count, limbs_buffer, rma.allocator);
        }
        rma.metadata = m.metadata;
    }

    pub fn ensureTwosCompCapacity(r: *Managed, bit_count: usize) !void {
        try r.ensureCapacity(calcTwosCompLimbCount(bit_count));
    }

    pub fn ensureAddScalarCapacity(r: *Managed, a: Const, scalar: anytype) !void {
        try r.ensureCapacity(@max(a.len(), calcLimbLen(scalar)) + 1);
    }

    pub fn ensureAddCapacity(r: *Managed, a: Const, b: Const) !void {
        try r.ensureCapacity(@max(a.len(), b.len()) + 1);
    }

    pub fn ensureMulCapacity(rma: *Managed, a: Const, b: Const) !void {
        try rma.ensureCapacity(a.len() + b.len() + 1);
    }

    /// q = a / b (rem r)
    ///
    /// a / b are floored (rounded towards 0).
    ///
    /// Returns an error if memory could not be allocated.
    pub fn divFloor(q: *Managed, r: *Managed, a: *const Managed, b: *const Managed) !void {
        try q.ensureCapacity(a.len());
        try r.ensureCapacity(b.len());
        var mq = q.toMutable();
        var mr = r.toMutable();
        const limbs_buffer = try q.allocator.alloc(Limb, calcDivLimbsBufferLen(a.len(), b.len()));
        defer q.allocator.free(limbs_buffer);
        mq.divFloor(&mr, a.toConst(), b.toConst(), limbs_buffer);
        q.metadata = mq.metadata;
        r.metadata = mr.metadata;
    }

    /// q = a / b (rem r)
    ///
    /// a / b are truncated (rounded towards -inf).
    ///
    /// Returns an error if memory could not be allocated.
    pub fn divTrunc(q: *Managed, r: *Managed, a: *const Managed, b: *const Managed) !void {
        try q.ensureCapacity(a.len());
        try r.ensureCapacity(b.len());
        var mq = q.toMutable();
        var mr = r.toMutable();
        const limbs_buffer = try q.allocator.alloc(Limb, calcDivLimbsBufferLen(a.len(), b.len()));
        defer q.allocator.free(limbs_buffer);
        mq.divTrunc(&mr, a.toConst(), b.toConst(), limbs_buffer);
        q.metadata = mq.metadata;
        r.metadata = mr.metadata;
    }

    /// r = a << shift, in other words, r = a * 2^shift
    /// r and a may alias.
    pub fn shiftLeft(r: *Managed, a: *const Managed, shift: usize) !void {
        try r.ensureCapacity(a.len() + (shift / limb_bits) + 1);
        var m = r.toMutable();
        m.shiftLeft(a.toConst(), shift);
        r.metadata = m.metadata;
    }

    /// r = a <<| shift with 2s-complement saturating semantics.
    /// r and a may alias.
    pub fn shiftLeftSat(r: *Managed, a: *const Managed, shift: usize, signedness: Signedness, bit_count: usize) !void {
        try r.ensureTwosCompCapacity(bit_count);
        var m = r.toMutable();
        m.shiftLeftSat(a.toConst(), shift, signedness, bit_count);
        r.metadata = m.metadata;
    }

    /// r = a >> shift
    /// r and a may alias.
    pub fn shiftRight(r: *Managed, a: *const Managed, shift: usize) !void {
        if (a.len() <= shift / limb_bits) {
            // Shifting negative numbers converges to -1 instead of 0
            if (a.sign() == .pos) {
                r.metadata = Metadata.init(.pos, 1);
                r.limbs[0] = 0;
            } else {
                r.metadata = Metadata.init(.neg, 1);
                r.limbs[0] = 1;
            }
            return;
        }

        try r.ensureCapacity(a.len() - (shift / limb_bits));
        var m = r.toMutable();
        m.shiftRight(a.toConst(), shift);
        r.metadata = m.metadata;
    }

    /// r = ~a under 2s-complement wrapping semantics.
    /// r and a may alias.
    pub fn bitNotWrap(r: *Managed, a: *const Managed, signedness: Signedness, bit_count: usize) !void {
        try r.ensureTwosCompCapacity(bit_count);
        var m = r.toMutable();
        m.bitNotWrap(a.toConst(), signedness, bit_count);
        r.metadata = m.metadata;
    }

    /// r = a | b
    ///
    /// a and b are zero-extended to the longer of a or b.
    pub fn bitOr(r: *Managed, a: *const Managed, b: *const Managed) !void {
        try r.ensureCapacity(@max(a.len(), b.len()));
        var m = r.toMutable();
        m.bitOr(a.toConst(), b.toConst());
        r.metadata = m.metadata;
    }

    /// r = a & b
    pub fn bitAnd(r: *Managed, a: *const Managed, b: *const Managed) !void {
        const cap = if (a.sign() == .pos or b.sign() == .pos)
            @min(a.len(), b.len())
        else
            @max(a.len(), b.len()) + 1;
        try r.ensureCapacity(cap);
        var m = r.toMutable();
        m.bitAnd(a.toConst(), b.toConst());
        r.metadata = m.metadata;
    }

    /// r = a ^ b
    pub fn bitXor(r: *Managed, a: *const Managed, b: *const Managed) !void {
        const cap = @max(a.len(), b.len()) + @intFromBool(a.sign() != b.sign());
        try r.ensureCapacity(cap);

        var m = r.toMutable();
        m.bitXor(a.toConst(), b.toConst());
        r.metadata = m.metadata;
    }

    /// rma may alias x or y.
    /// x and y may alias each other.
    ///
    /// rma's allocator is used for temporary storage to boost multiplication performance.
    pub fn gcd(rma: *Managed, x: *const Managed, y: *const Managed) !void {
        try rma.ensureCapacity(@min(x.len(), y.len()));
        var m = rma.toMutable();
        var limbs_buffer = std.ArrayList(Limb).init(rma.allocator);
        defer limbs_buffer.deinit();
        try m.gcd(x.toConst(), y.toConst(), &limbs_buffer);
        rma.metadata = m.metadata;
    }

    /// r = a * a
    pub fn sqr(rma: *Managed, a: *const Managed) !void {
        const needed_limbs = 2 * a.len() + 1;

        if (rma.limbs.ptr == a.limbs.ptr) {
            var m = try Managed.initCapacity(rma.allocator, needed_limbs);
            errdefer m.deinit();
            var m_mut = m.toMutable();
            m_mut.sqrNoAlias(a.toConst(), rma.allocator);
            m.metadata = m_mut.metadata;

            rma.deinit();
            rma.swap(&m);
        } else {
            try rma.ensureCapacity(needed_limbs);
            var rma_mut = rma.toMutable();
            rma_mut.sqrNoAlias(a.toConst(), rma.allocator);
            rma.metadata = rma_mut.metadata;
        }
    }

    pub fn pow(rma: *Managed, a: *const Managed, b: u32) !void {
        const needed_limbs = calcPowLimbsBufferLen(a.bitCountAbs(), b);

        const limbs_buffer = try rma.allocator.alloc(Limb, needed_limbs);
        defer rma.allocator.free(limbs_buffer);

        if (rma.limbs.ptr == a.limbs.ptr) {
            var m = try Managed.initCapacity(rma.allocator, needed_limbs);
            errdefer m.deinit();
            var m_mut = m.toMutable();
            m_mut.pow(a.toConst(), b, limbs_buffer);

            rma.deinit();
            rma.swap(&m);
        } else {
            try rma.ensureCapacity(needed_limbs);
            var rma_mut = rma.toMutable();
            rma_mut.pow(a.toConst(), b, limbs_buffer);
            rma.metadata = rma_mut.metadata;
        }
    }

    /// r = ⌊√a⌋
    pub fn sqrt(rma: *Managed, a: *const Managed) !void {
        const bit_count = a.bitCountAbs();

        if (bit_count == 0) {
            try rma.set(0);
            rma.setMetadata(a.isPositive(), rma.len());
            return;
        }

        if (!a.isPositive()) {
            return error.SqrtOfNegativeNumber;
        }

        const needed_limbs = calcSqrtLimbsBufferLen(bit_count);
        const limbs_buffer = try rma.allocator.alloc(Limb, needed_limbs);
        defer rma.allocator.free(limbs_buffer);

        try rma.ensureCapacity((a.len() - 1) / 2 + 1);
        var m = rma.toMutable();
        m.sqrt(a.toConst(), limbs_buffer);
        rma.metadata = m.metadata;
    }

    /// r = truncate(Int(signedness, bit_count), a)
    pub fn truncate(r: *Managed, a: *const Managed, signedness: Signedness, bit_count: usize) !void {
        try r.ensureCapacity(calcTwosCompLimbCount(bit_count));
        var m = r.toMutable();
        m.truncate(a.toConst(), signedness, bit_count);
        r.metadata = m.metadata;
    }

    /// r = saturate(Int(signedness, bit_count), a)
    pub fn saturate(r: *Managed, a: *const Managed, signedness: Signedness, bit_count: usize) !void {
        try r.ensureCapacity(calcTwosCompLimbCount(bit_count));
        var m = r.toMutable();
        m.saturate(a.toConst(), signedness, bit_count);
        r.metadata = m.metadata;
    }

    /// r = @popCount(a) with 2s-complement semantics.
    /// r and a may be aliases.
    pub fn popCount(r: *Managed, a: *const Managed, bit_count: usize) !void {
        try r.ensureCapacity(calcTwosCompLimbCount(bit_count));
        var m = r.toMutable();
        m.popCount(a.toConst(), bit_count);
        r.metadata = m.metadata;
    }
};

/// Different operators which can be used in accumulation style functions
/// (llmulacc, llmulaccKaratsuba, llmulaccLong, llmulLimb). In all these functions,
/// a computed value is accumulated with an existing result.
const AccOp = enum {
    /// The computed value is added to the result.
    add,

    /// The computed value is subtracted from the result.
    sub,
};

/// Knuth 4.3.1, Algorithm M.
///
/// r = r (op) a * b
/// r MUST NOT alias any of a or b.
///
/// The result is computed modulo `r.len`. When `r.len >= a.len + b.len`, no overflow occurs.
fn llmulacc(comptime op: AccOp, opt_allocator: ?Allocator, r: []Limb, a: []const Limb, b: []const Limb) void {
    @setRuntimeSafety(debug_safety);
    assert(r.len >= a.len);
    assert(r.len >= b.len);

    // Order greatest first.
    var x = a;
    var y = b;
    if (a.len < b.len) {
        x = b;
        y = a;
    }

    k_mul: {
        if (y.len > 48) {
            if (opt_allocator) |allocator| {
                llmulaccKaratsuba(op, allocator, r, x, y) catch |err| switch (err) {
                    error.OutOfMemory => break :k_mul, // handled below
                };
                return;
            }
        }
    }

    llmulaccLong(op, r, x, y);
}

/// Knuth 4.3.1, Algorithm M.
///
/// r = r (op) a * b
/// r MUST NOT alias any of a or b.
///
/// The result is computed modulo `r.len`. When `r.len >= a.len + b.len`, no overflow occurs.
fn llmulaccKaratsuba(
    comptime op: AccOp,
    allocator: Allocator,
    r: []Limb,
    a: []const Limb,
    b: []const Limb,
) error{OutOfMemory}!void {
    @setRuntimeSafety(debug_safety);
    assert(r.len >= a.len);
    assert(a.len >= b.len);

    // Classical karatsuba algorithm:
    // a = a1 * B + a0
    // b = b1 * B + b0
    // Where a0, b0 < B
    //
    // We then have:
    // ab = a * b
    //    = (a1 * B + a0) * (b1 * B + b0)
    //    = a1 * b1 * B * B + a1 * B * b0 + a0 * b1 * B + a0 * b0
    //    = a1 * b1 * B * B + (a1 * b0 + a0 * b1) * B + a0 * b0
    //
    // Note that:
    // a1 * b0 + a0 * b1
    //    = (a1 + a0)(b1 + b0) - a1 * b1 - a0 * b0
    //    = (a0 - a1)(b1 - b0) + a1 * b1 + a0 * b0
    //
    // This yields:
    // ab = p2 * B^2 + (p0 + p1 + p2) * B + p0
    //
    // Where:
    // p0 = a0 * b0
    // p1 = (a0 - a1)(b1 - b0)
    // p2 = a1 * b1
    //
    // Note, (a0 - a1) and (b1 - b0) produce values -B < x < B, and so we need to mind the sign here.
    // We also have:
    // 0 <= p0 <= 2B
    // -2B <= p1 <= 2B
    //
    // Note, when B is a multiple of the limb size, multiplies by B amount to shifts or
    // slices of a limbs array.
    //
    // This function computes the result of the multiplication modulo r.len. This means:
    // - p2 and p1 only need to be computed modulo r.len - B.
    // - In the case of p2, p2 * B^2 needs to be added modulo r.len - 2 * B.

    const split = b.len / 2; // B

    const limbs_after_split = r.len - split; // Limbs to compute for p1 and p2.
    const limbs_after_split2 = r.len - split * 2; // Limbs to add for p2 * B^2.

    // For a0 and b0 we need the full range.
    const a0 = a[0..llnormalize(a[0..split])];
    const b0 = b[0..llnormalize(b[0..split])];

    // For a1 and b1 we only need `limbs_after_split` limbs.
    const a1 = blk: {
        var a1 = a[split..];
        a1.len = @min(llnormalize(a1), limbs_after_split);
        break :blk a1;
    };

    const b1 = blk: {
        var b1 = b[split..];
        b1.len = @min(llnormalize(b1), limbs_after_split);
        break :blk b1;
    };

    // Note that the above slices relative to `split` work because we have a.len > b.len.

    // We need some temporary memory to store intermediate results.
    // Note, we can reduce the amount of temporaries we need by reordering the computation here:
    // ab = p2 * B^2 + (p0 + p1 + p2) * B + p0
    //    = p2 * B^2 + (p0 * B + p1 * B + p2 * B) + p0
    //    = (p2 * B^2 + p2 * B) + (p0 * B + p0) + p1 * B

    // Allocate at least enough memory to be able to multiply the upper two segments of a and b, assuming
    // no overflow.
    const tmp = try allocator.alloc(Limb, a.len - split + b.len - split);
    defer allocator.free(tmp);

    // Compute p2.
    // Note, we don't need to compute all of p2, just enough limbs to satisfy r.
    const p2_limbs = @min(limbs_after_split, a1.len + b1.len);

    @memset(tmp[0..p2_limbs], 0);
    llmulacc(.add, allocator, tmp[0..p2_limbs], a1[0..@min(a1.len, p2_limbs)], b1[0..@min(b1.len, p2_limbs)]);
    const p2 = tmp[0..llnormalize(tmp[0..p2_limbs])];

    // Add p2 * B to the result.
    llaccum(op, r[split..], p2);

    // Add p2 * B^2 to the result if required.
    if (limbs_after_split2 > 0) {
        llaccum(op, r[split * 2 ..], p2[0..@min(p2.len, limbs_after_split2)]);
    }

    // Compute p0.
    // Since a0.len, b0.len <= split and r.len >= split * 2, the full width of p0 needs to be computed.
    const p0_limbs = a0.len + b0.len;
    @memset(tmp[0..p0_limbs], 0);
    llmulacc(.add, allocator, tmp[0..p0_limbs], a0, b0);
    const p0 = tmp[0..llnormalize(tmp[0..p0_limbs])];

    // Add p0 to the result.
    llaccum(op, r, p0);

    // Add p0 * B to the result. In this case, we may not need all of it.
    llaccum(op, r[split..], p0[0..@min(limbs_after_split, p0.len)]);

    // Finally, compute and add p1.
    // From now on we only need `limbs_after_split` limbs for a0 and b0, since the result of the
    // following computation will be added * B.
    const a0x = a0[0..@min(a0.len, limbs_after_split)];
    const b0x = b0[0..@min(b0.len, limbs_after_split)];

    const j0_sign = llcmp(a0x, a1);
    const j1_sign = llcmp(b1, b0x);

    if (j0_sign * j1_sign == 0) {
        // p1 is zero, we don't need to do any computation at all.
        return;
    }

    @memset(tmp, 0);

    // p1 is nonzero, so compute the intermediary terms j0 = a0 - a1 and j1 = b1 - b0.
    // Note that in this case, we again need some storage for intermediary results
    // j0 and j1. Since we have tmp.len >= 2B, we can store both
    // intermediaries in the already allocated array.
    const j0 = tmp[0 .. a.len - split];
    const j1 = tmp[a.len - split ..];

    // Ensure that no subtraction overflows.
    if (j0_sign == 1) {
        // a0 > a1.
        _ = llsubcarry(j0, a0x, a1);
    } else {
        // a0 < a1.
        _ = llsubcarry(j0, a1, a0x);
    }

    if (j1_sign == 1) {
        // b1 > b0.
        _ = llsubcarry(j1, b1, b0x);
    } else {
        // b1 > b0.
        _ = llsubcarry(j1, b0x, b1);
    }

    if (j0_sign * j1_sign == 1) {
        // If j0 and j1 are both positive, we now have:
        // p1 = j0 * j1
        // If j0 and j1 are both negative, we now have:
        // p1 = -j0 * -j1 = j0 * j1
        // In this case we can add p1 to the result using llmulacc.
        llmulacc(op, allocator, r[split..], j0[0..llnormalize(j0)], j1[0..llnormalize(j1)]);
    } else {
        // In this case either j0 or j1 is negative, an we have:
        // p1 = -(j0 * j1)
        // Now we need to subtract instead of accumulate.
        const inverted_op = if (op == .add) .sub else .add;
        llmulacc(inverted_op, allocator, r[split..], j0[0..llnormalize(j0)], j1[0..llnormalize(j1)]);
    }
}

/// r = r (op) a.
/// The result is computed modulo `r.len`.
fn llaccum(comptime op: AccOp, r: []Limb, a: []const Limb) void {
    @setRuntimeSafety(debug_safety);
    if (op == .sub) {
        _ = llsubcarry(r, r, a);
        return;
    }

    assert(r.len != 0 and a.len != 0);
    assert(r.len >= a.len);

    var i: usize = 0;
    var carry: Limb = 0;

    while (i < a.len) : (i += 1) {
        const ov1 = @addWithOverflow(r[i], a[i]);
        r[i] = ov1[0];
        const ov2 = @addWithOverflow(r[i], carry);
        r[i] = ov2[0];
        carry = @as(Limb, ov1[1]) + ov2[1];
    }

    while ((carry != 0) and i < r.len) : (i += 1) {
        const ov = @addWithOverflow(r[i], carry);
        r[i] = ov[0];
        carry = ov[1];
    }
}

/// Returns -1, 0, 1 if |a| < |b|, |a| == |b| or |a| > |b| respectively for limbs.
pub fn llcmp(a: []const Limb, b: []const Limb) i8 {
    @setRuntimeSafety(debug_safety);
    const a_len = llnormalize(a);
    const b_len = llnormalize(b);
    if (a_len < b_len) {
        return -1;
    }
    if (a_len > b_len) {
        return 1;
    }

    var i: usize = a_len - 1;
    while (i != 0) : (i -= 1) {
        if (a[i] != b[i]) {
            break;
        }
    }

    if (a[i] < b[i]) {
        return -1;
    } else if (a[i] > b[i]) {
        return 1;
    } else {
        return 0;
    }
}

/// r = r (op) y * xi
/// The result is computed modulo `r.len`. When `r.len >= a.len + b.len`, no overflow occurs.
fn llmulaccLong(comptime op: AccOp, r: []Limb, a: []const Limb, b: []const Limb) void {
    @setRuntimeSafety(debug_safety);
    assert(r.len >= a.len);
    assert(a.len >= b.len);

    var i: usize = 0;
    while (i < b.len) : (i += 1) {
        _ = llmulLimb(op, r[i..], a, b[i]);
    }
}

/// r = r (op) y * xi
/// The result is computed modulo `r.len`.
/// Returns whether the operation overflowed.
fn llmulLimb(comptime op: AccOp, acc: []Limb, y: []const Limb, xi: Limb) bool {
    @setRuntimeSafety(debug_safety);
    if (xi == 0) {
        return false;
    }

    const split = @min(y.len, acc.len);
    var a_lo = acc[0..split];
    var a_hi = acc[split..];

    switch (op) {
        .add => {
            var carry: Limb = 0;
            var j: usize = 0;
            while (j < a_lo.len) : (j += 1) {
                a_lo[j] = addMulLimbWithCarry(a_lo[j], y[j], xi, &carry);
            }

            j = 0;
            while ((carry != 0) and (j < a_hi.len)) : (j += 1) {
                const ov = @addWithOverflow(a_hi[j], carry);
                a_hi[j] = ov[0];
                carry = ov[1];
            }

            return carry != 0;
        },
        .sub => {
            var borrow: Limb = 0;
            var j: usize = 0;
            while (j < a_lo.len) : (j += 1) {
                a_lo[j] = subMulLimbWithBorrow(a_lo[j], y[j], xi, &borrow);
            }

            j = 0;
            while ((borrow != 0) and (j < a_hi.len)) : (j += 1) {
                const ov = @subWithOverflow(a_hi[j], borrow);
                a_hi[j] = ov[0];
                borrow = ov[1];
            }

            return borrow != 0;
        },
    }
}

/// returns the min length the limb could be.
fn llnormalize(a: []const Limb) usize {
    @setRuntimeSafety(debug_safety);
    var j = a.len;
    while (j > 0) : (j -= 1) {
        if (a[j - 1] != 0) {
            break;
        }
    }

    // Handle zero
    return if (j != 0) j else 1;
}

/// Knuth 4.3.1, Algorithm S.
fn llsubcarry(r: []Limb, a: []const Limb, b: []const Limb) Limb {
    @setRuntimeSafety(debug_safety);
    assert(a.len != 0 and b.len != 0);
    assert(a.len >= b.len);
    assert(r.len >= a.len);

    var i: usize = 0;
    var borrow: Limb = 0;

    while (i < b.len) : (i += 1) {
        const ov1 = @subWithOverflow(a[i], b[i]);
        r[i] = ov1[0];
        const ov2 = @subWithOverflow(r[i], borrow);
        r[i] = ov2[0];
        borrow = @as(Limb, ov1[1]) + ov2[1];
    }

    while (i < a.len) : (i += 1) {
        const ov = @subWithOverflow(a[i], borrow);
        r[i] = ov[0];
        borrow = ov[1];
    }

    return borrow;
}

fn llsub(r: []Limb, a: []const Limb, b: []const Limb) void {
    @setRuntimeSafety(debug_safety);
    assert(a.len > b.len or (a.len == b.len and a[a.len - 1] >= b[b.len - 1]));
    assert(llsubcarry(r, a, b) == 0);
}

/// Knuth 4.3.1, Algorithm A.
fn lladdcarry(r: []Limb, a: []const Limb, b: []const Limb) Limb {
    @setRuntimeSafety(debug_safety);
    assert(a.len != 0 and b.len != 0);
    assert(a.len >= b.len);
    assert(r.len >= a.len);

    var i: usize = 0;
    var carry: Limb = 0;

    while (i < b.len) : (i += 1) {
        const ov1 = @addWithOverflow(a[i], b[i]);
        r[i] = ov1[0];
        const ov2 = @addWithOverflow(r[i], carry);
        r[i] = ov2[0];
        carry = @as(Limb, ov1[1]) + ov2[1];
    }

    while (i < a.len) : (i += 1) {
        const ov = @addWithOverflow(a[i], carry);
        r[i] = ov[0];
        carry = ov[1];
    }

    return carry;
}

fn lladd(r: []Limb, a: []const Limb, b: []const Limb) void {
    @setRuntimeSafety(debug_safety);
    assert(r.len >= a.len + 1);
    r[a.len] = lladdcarry(r, a, b);
}

/// Knuth 4.3.1, Exercise 16.
fn lldiv1(quo: []Limb, rem: *Limb, a: []const Limb, b: Limb) void {
    @setRuntimeSafety(debug_safety);
    assert(a.len > 1 or a[0] >= b);
    assert(quo.len >= a.len);

    rem.* = 0;
    for (a, 0..) |_, ri| {
        const i = a.len - ri - 1;
        const pdiv = ((@as(DoubleLimb, rem.*) << limb_bits) | a[i]);

        if (pdiv == 0) {
            quo[i] = 0;
            rem.* = 0;
        } else if (pdiv < b) {
            quo[i] = 0;
            rem.* = @as(Limb, @truncate(pdiv));
        } else if (pdiv == b) {
            quo[i] = 1;
            rem.* = 0;
        } else {
            quo[i] = @as(Limb, @truncate(@divTrunc(pdiv, b)));
            rem.* = @as(Limb, @truncate(pdiv - (quo[i] *% b)));
        }
    }
}

fn lldiv0p5(quo: []Limb, rem: *Limb, a: []const Limb, b: HalfLimb) void {
    @setRuntimeSafety(debug_safety);
    assert(a.len > 1 or a[0] >= b);
    assert(quo.len >= a.len);

    rem.* = 0;
    for (a, 0..) |_, ri| {
        const i = a.len - ri - 1;
        const ai_high = a[i] >> half_limb_bits;
        const ai_low = a[i] & ((1 << half_limb_bits) - 1);

        // Split the division into two divisions acting on half a limb each. Carry remainder.
        const ai_high_with_carry = (rem.* << half_limb_bits) | ai_high;
        const ai_high_quo = ai_high_with_carry / b;
        rem.* = ai_high_with_carry % b;

        const ai_low_with_carry = (rem.* << half_limb_bits) | ai_low;
        const ai_low_quo = ai_low_with_carry / b;
        rem.* = ai_low_with_carry % b;

        quo[i] = (ai_high_quo << half_limb_bits) | ai_low_quo;
    }
}

fn llshl(r: []Limb, a: []const Limb, shift: usize) void {
    @setRuntimeSafety(debug_safety);
    assert(a.len >= 1);

    const interior_limb_shift = @as(Log2Limb, @truncate(shift));

    // We only need the extra limb if the shift of the last element overflows.
    // This is useful for the implementation of `shiftLeftSat`.
    if (a[a.len - 1] << interior_limb_shift >> interior_limb_shift != a[a.len - 1]) {
        assert(r.len >= a.len + (shift / limb_bits) + 1);
    } else {
        assert(r.len >= a.len + (shift / limb_bits));
    }

    const limb_shift = shift / limb_bits + 1;

    var carry: Limb = 0;
    var i: usize = 0;
    while (i < a.len) : (i += 1) {
        const src_i = a.len - i - 1;
        const dst_i = src_i + limb_shift;

        const src_digit = a[src_i];
        r[dst_i] = carry | @call(.always_inline, math.shr, .{
            Limb,
            src_digit,
            limb_bits - @as(Limb, @intCast(interior_limb_shift)),
        });
        carry = (src_digit << interior_limb_shift);
    }

    r[limb_shift - 1] = carry;
    @memset(r[0 .. limb_shift - 1], 0);
}

fn llshr(r: []Limb, a: []const Limb, shift: usize) void {
    @setRuntimeSafety(debug_safety);
    assert(a.len >= 1);
    assert(r.len >= a.len - (shift / limb_bits));

    const limb_shift = shift / limb_bits;
    const interior_limb_shift = @as(Log2Limb, @truncate(shift));

    var carry: Limb = 0;
    var i: usize = 0;
    while (i < a.len - limb_shift) : (i += 1) {
        const src_i = a.len - i - 1;
        const dst_i = src_i - limb_shift;

        const src_digit = a[src_i];
        r[dst_i] = carry | (src_digit >> interior_limb_shift);
        carry = @call(.always_inline, math.shl, .{
            Limb,
            src_digit,
            limb_bits - @as(Limb, @intCast(interior_limb_shift)),
        });
    }
}

// r = ~r
fn llnot(r: []Limb) void {
    @setRuntimeSafety(debug_safety);

    for (r) |*elem| {
        elem.* = ~elem.*;
    }
}

// r = a | b with 2s complement semantics.
// r may alias.
// a and b must not be 0.
// Returns `true` when the result is positive.
// When b is positive, r requires at least `a.len` limbs of storage.
// When b is negative, r requires at least `b.len` limbs of storage.
fn llsignedor(r: []Limb, a: []const Limb, a_positive: bool, b: []const Limb, b_positive: bool) bool {
    @setRuntimeSafety(debug_safety);
    assert(r.len >= a.len);
    assert(a.len >= b.len);

    if (a_positive and b_positive) {
        // Trivial case, result is positive.
        var i: usize = 0;
        while (i < b.len) : (i += 1) {
            r[i] = a[i] | b[i];
        }
        while (i < a.len) : (i += 1) {
            r[i] = a[i];
        }

        return true;
    } else if (!a_positive and b_positive) {
        // Result is negative.
        // r = (--a) | b
        //   = ~(-a - 1) | b
        //   = ~(-a - 1) | ~~b
        //   = ~((-a - 1) & ~b)
        //   = -(((-a - 1) & ~b) + 1)

        var i: usize = 0;
        var a_borrow: u1 = 1;
        var r_carry: u1 = 1;

        while (i < b.len) : (i += 1) {
            const ov1 = @subWithOverflow(a[i], a_borrow);
            a_borrow = ov1[1];
            const ov2 = @addWithOverflow(ov1[0] & ~b[i], r_carry);
            r[i] = ov2[0];
            r_carry = ov2[1];
        }

        // In order for r_carry to be nonzero at this point, ~b[i] would need to be
        // all ones, which would require b[i] to be zero. This cannot be when
        // b is normalized, so there cannot be a carry here.
        // Also, x & ~b can only clear bits, so (x & ~b) <= x, meaning (-a - 1) + 1 never overflows.
        assert(r_carry == 0);

        // With b = 0, we get (-a - 1) & ~0 = -a - 1.
        // Note, if a_borrow is zero we do not need to compute anything for
        // the higher limbs so we can early return here.
        while (i < a.len and a_borrow == 1) : (i += 1) {
            const ov = @subWithOverflow(a[i], a_borrow);
            r[i] = ov[0];
            a_borrow = ov[1];
        }

        assert(a_borrow == 0); // a was 0.

        return false;
    } else if (a_positive and !b_positive) {
        // Result is negative.
        // r = a | (--b)
        //   = a | ~(-b - 1)
        //   = ~~a | ~(-b - 1)
        //   = ~(~a & (-b - 1))
        //   = -((~a & (-b - 1)) + 1)

        var i: usize = 0;
        var b_borrow: u1 = 1;
        var r_carry: u1 = 1;

        while (i < b.len) : (i += 1) {
            const ov1 = @subWithOverflow(b[i], b_borrow);
            b_borrow = ov1[1];
            const ov2 = @addWithOverflow(~a[i] & ov1[0], r_carry);
            r[i] = ov2[0];
            r_carry = ov2[1];
        }

        // b is at least 1, so this should never underflow.
        assert(b_borrow == 0); // b was 0

        // x & ~a can only clear bits, so (x & ~a) <= x, meaning (-b - 1) + 1 never overflows.
        assert(r_carry == 0);

        // With b = 0 and b_borrow = 0, we get ~a & (-0 - 0) = ~a & 0 = 0.
        // Omit setting the upper bytes, just deal with those when calling llsignedor.

        return false;
    } else {
        // Result is negative.
        // r = (--a) | (--b)
        //   = ~(-a - 1) | ~(-b - 1)
        //   = ~((-a - 1) & (-b - 1))
        //   = -(~(~((-a - 1) & (-b - 1))) + 1)
        //   = -((-a - 1) & (-b - 1) + 1)

        var i: usize = 0;
        var a_borrow: u1 = 1;
        var b_borrow: u1 = 1;
        var r_carry: u1 = 1;

        while (i < b.len) : (i += 1) {
            const ov1 = @subWithOverflow(a[i], a_borrow);
            a_borrow = ov1[1];
            const ov2 = @subWithOverflow(b[i], b_borrow);
            b_borrow = ov2[1];
            const ov3 = @addWithOverflow(ov1[0] & ov2[0], r_carry);
            r[i] = ov3[0];
            r_carry = ov3[1];
        }

        // b is at least 1, so this should never underflow.
        assert(b_borrow == 0); // b was 0

        // Can never overflow because in order for b_limb to be maxInt(Limb),
        // b_borrow would need to equal 1.

        // x & y can only clear bits, meaning x & y <= x and x & y <= y. This implies that
        // for x = a - 1 and y = b - 1, the +1 term would never cause an overflow.
        assert(r_carry == 0);

        // With b = 0 and b_borrow = 0 we get (-a - 1) & (-0 - 0) = (-a - 1) & 0 = 0.
        // Omit setting the upper bytes, just deal with those when calling llsignedor.
        return false;
    }
}

// r = a & b with 2s complement semantics.
// r may alias.
// a and b must not be 0.
// Returns `true` when the result is positive.
// When either or both of a and b are positive, r requires at least `b.len` limbs of storage.
// When both a and b are negative, r requires at least `a.limbs.len + 1` limbs of storage.
fn llsignedand(r: []Limb, a: []const Limb, a_positive: bool, b: []const Limb, b_positive: bool) bool {
    @setRuntimeSafety(debug_safety);
    assert(a.len != 0 and b.len != 0);
    assert(a.len >= b.len);
    assert(r.len >= if (!a_positive and !b_positive) a.len + 1 else b.len);

    if (a_positive and b_positive) {
        // Trivial case, result is positive.
        var i: usize = 0;
        while (i < b.len) : (i += 1) {
            r[i] = a[i] & b[i];
        }

        // With b = 0 we have a & 0 = 0, so the upper bytes are zero.
        // Omit setting them here and simply discard them whenever
        // llsignedand is called.

        return true;
    } else if (!a_positive and b_positive) {
        // Result is positive.
        // r = (--a) & b
        //   = ~(-a - 1) & b

        var i: usize = 0;
        var a_borrow: u1 = 1;

        while (i < b.len) : (i += 1) {
            const ov = @subWithOverflow(a[i], a_borrow);
            a_borrow = ov[1];
            r[i] = ~ov[0] & b[i];
        }

        // With b = 0 we have ~(a - 1) & 0 = 0, so the upper bytes are zero.
        // Omit setting them here and simply discard them whenever
        // llsignedand is called.

        return true;
    } else if (a_positive and !b_positive) {
        // Result is positive.
        // r = a & (--b)
        //   = a & ~(-b - 1)

        var i: usize = 0;
        var b_borrow: u1 = 1;

        while (i < b.len) : (i += 1) {
            const ov = @subWithOverflow(b[i], b_borrow);
            b_borrow = ov[1];
            r[i] = a[i] & ~ov[0];
        }

        assert(b_borrow == 0); // b was 0

        // With b = 0 and b_borrow = 0 we have a & ~(-0 - 0) = a & 0 = 0, so
        // the upper bytes are zero.  Omit setting them here and simply discard
        // them whenever llsignedand is called.

        return true;
    } else {
        // Result is negative.
        // r = (--a) & (--b)
        //   = ~(-a - 1) & ~(-b - 1)
        //   = ~((-a - 1) | (-b - 1))
        //   = -(((-a - 1) | (-b - 1)) + 1)

        var i: usize = 0;
        var a_borrow: u1 = 1;
        var b_borrow: u1 = 1;
        var r_carry: u1 = 1;

        while (i < b.len) : (i += 1) {
            const ov1 = @subWithOverflow(a[i], a_borrow);
            a_borrow = ov1[1];
            const ov2 = @subWithOverflow(b[i], b_borrow);
            b_borrow = ov2[1];
            const ov3 = @addWithOverflow(ov1[0] | ov2[0], r_carry);
            r[i] = ov3[0];
            r_carry = ov3[1];
        }

        // b is at least 1, so this should never underflow.
        assert(b_borrow == 0); // b was 0

        // With b = 0 and b_borrow = 0 we get (-a - 1) | (-0 - 0) = (-a - 1) | 0 = -a - 1.
        while (i < a.len) : (i += 1) {
            const ov1 = @subWithOverflow(a[i], a_borrow);
            a_borrow = ov1[1];
            const ov2 = @addWithOverflow(ov1[0], r_carry);
            r[i] = ov2[0];
            r_carry = ov2[1];
        }

        assert(a_borrow == 0); // a was 0.

        // The final addition can overflow here, so we need to keep that in mind.
        r[i] = r_carry;

        return false;
    }
}

// r = a ^ b with 2s complement semantics.
// r may alias.
// a and b must not be -0.
// Returns `true` when the result is positive.
// If the sign of a and b is equal, then r requires at least `@max(a.len, b.len)` limbs are required.
// Otherwise, r requires at least `@max(a.len, b.len) + 1` limbs.
fn llsignedxor(r: []Limb, a: []const Limb, a_positive: bool, b: []const Limb, b_positive: bool) bool {
    @setRuntimeSafety(debug_safety);
    assert(a.len != 0 and b.len != 0);
    assert(r.len >= a.len);
    assert(a.len >= b.len);

    // If a and b are positive, the result is positive and r = a ^ b.
    // If a negative, b positive, result is negative and we have
    // r = --(--a ^ b)
    //   = --(~(-a - 1) ^ b)
    //   = -(~(~(-a - 1) ^ b) + 1)
    //   = -(((-a - 1) ^ b) + 1)
    // Same if a is positive and b is negative, sides switched.
    // If both a and b are negative, the result is positive and we have
    // r = (--a) ^ (--b)
    //   = ~(-a - 1) ^ ~(-b - 1)
    //   = (-a - 1) ^ (-b - 1)
    // These operations can be made more generic as follows:
    // - If a is negative, subtract 1 from |a| before the xor.
    // - If b is negative, subtract 1 from |b| before the xor.
    // - if the result is supposed to be negative, add 1.

    var i: usize = 0;
    var a_borrow = @intFromBool(!a_positive);
    var b_borrow = @intFromBool(!b_positive);
    var r_carry = @intFromBool(a_positive != b_positive);

    while (i < b.len) : (i += 1) {
        const ov1 = @subWithOverflow(a[i], a_borrow);
        a_borrow = ov1[1];
        const ov2 = @subWithOverflow(b[i], b_borrow);
        b_borrow = ov2[1];
        const ov3 = @addWithOverflow(ov1[0] ^ ov2[0], r_carry);
        r[i] = ov3[0];
        r_carry = ov3[1];
    }

    while (i < a.len) : (i += 1) {
        const ov1 = @subWithOverflow(a[i], a_borrow);
        a_borrow = ov1[1];
        const ov2 = @addWithOverflow(ov1[0], r_carry);
        r[i] = ov2[0];
        r_carry = ov2[1];
    }

    // If both inputs don't share the same sign, an extra limb is required.
    if (a_positive != b_positive) {
        r[i] = r_carry;
    } else {
        assert(r_carry == 0);
    }

    assert(a_borrow == 0);
    assert(b_borrow == 0);

    return a_positive == b_positive;
}

/// r MUST NOT alias x.
fn llsquareBasecase(r: []Limb, x: []const Limb) void {
    @setRuntimeSafety(debug_safety);

    const x_norm = x;
    assert(r.len >= 2 * x_norm.len + 1);

    // Compute the square of a N-limb bigint with only (N^2 + N)/2
    // multiplications by exploiting the symmetry of the coefficients around the
    // diagonal:
    //
    //           a   b   c *
    //           a   b   c =
    // -------------------
    //          ca  cb  cc +
    //      ba  bb  bc     +
    //  aa  ab  ac
    //
    // Note that:
    //  - Each mixed-product term appears twice for each column,
    //  - Squares are always in the 2k (0 <= k < N) column

    for (x_norm, 0..) |v, i| {
        // Accumulate all the x[i]*x[j] (with x!=j) products
        const overflow = llmulLimb(.add, r[2 * i + 1 ..], x_norm[i + 1 ..], v);
        assert(!overflow);
    }

    // Each product appears twice, multiply by 2
    llshl(r, r[0 .. 2 * x_norm.len], 1);

    for (x_norm, 0..) |v, i| {
        // Compute and add the squares
        const overflow = llmulLimb(.add, r[2 * i ..], x[i..][0..1], v);
        assert(!overflow);
    }
}

/// Knuth 4.6.3
fn llpow(r: []Limb, a: []const Limb, b: u32, tmp_limbs: []Limb) void {
    var tmp1: []Limb = undefined;
    var tmp2: []Limb = undefined;

    // Multiplication requires no aliasing between the operand and the result
    // variable, use the output limbs and another temporary set to overcome this
    // limitation.
    // The initial assignment makes the result end in `r` so an extra memory
    // copy is saved, each 1 flips the index twice so it's only the zeros that
    // matter.
    const b_leading_zeros = @clz(b);
    const exp_zeros = @popCount(~b) - b_leading_zeros;
    if (exp_zeros & 1 != 0) {
        tmp1 = tmp_limbs;
        tmp2 = r;
    } else {
        tmp1 = r;
        tmp2 = tmp_limbs;
    }

    @memcpy(tmp1[0..a.len], a);
    @memset(tmp1[a.len..], 0);

    // Scan the exponent as a binary number, from left to right, dropping the
    // most significant bit set.
    // Square the result if the current bit is zero, square and multiply by a if
    // it is one.
    const exp_bits = 32 - 1 - b_leading_zeros;
    var exp = b << @as(u5, @intCast(1 + b_leading_zeros));

    var i: usize = 0;
    while (i < exp_bits) : (i += 1) {
        // Square
        @memset(tmp2, 0);
        llsquareBasecase(tmp2, tmp1[0..llnormalize(tmp1)]);
        mem.swap([]Limb, &tmp1, &tmp2);
        // Multiply by a
        const ov = @shlWithOverflow(exp, 1);
        exp = ov[0];
        if (ov[1] != 0) {
            @memset(tmp2, 0);
            llmulacc(.add, null, tmp2, tmp1[0..llnormalize(tmp1)], a);
            mem.swap([]Limb, &tmp1, &tmp2);
        }
    }
}

/// a + b * c + *carry, sets carry to the overflow bits
pub fn addMulLimbWithCarry(a: Limb, b: Limb, c: Limb, carry: *Limb) Limb {
    @setRuntimeSafety(debug_safety);

    // ov1[0] = a + *carry
    const ov1 = @addWithOverflow(a, carry.*);

    // r2 = b * c
    const bc = @as(DoubleLimb, math.mulWide(Limb, b, c));
    const r2 = @as(Limb, @truncate(bc));
    const c2 = @as(Limb, @truncate(bc >> limb_bits));

    // ov2[0] = ov1[0] + r2
    const ov2 = @addWithOverflow(ov1[0], r2);

    // This never overflows, c1, c3 are either 0 or 1 and if both are 1 then
    // c2 is at least <= maxInt(Limb) - 2.
    carry.* = ov1[1] + c2 + ov2[1];

    return ov2[0];
}

/// a - b * c - *carry, sets carry to the overflow bits
fn subMulLimbWithBorrow(a: Limb, b: Limb, c: Limb, carry: *Limb) Limb {
    // ov1[0] = a - *carry
    const ov1 = @subWithOverflow(a, carry.*);

    // r2 = b * c
    const bc = @as(DoubleLimb, std.math.mulWide(Limb, b, c));
    const r2 = @as(Limb, @truncate(bc));
    const c2 = @as(Limb, @truncate(bc >> limb_bits));

    // ov2[0] = ov1[0] - r2
    const ov2 = @subWithOverflow(ov1[0], r2);
    carry.* = ov1[1] + c2 + ov2[1];

    return ov2[0];
}

// Storage must live for the lifetime of the returned value
fn fixedIntFromSignedDoubleLimb(A: SignedDoubleLimb, storage: []Limb) Mutable {
    assert(storage.len >= 2);

    const Au = @as(DoubleLimb, @intCast(if (A < 0) -A else A));
    storage[0] = @as(Limb, @truncate(Au));
    storage[1] = @as(Limb, @truncate(Au >> limb_bits));
    return .{
        .limbs = storage[0..2],
        .metadata = Metadata.init(Sign.fromBool(A >= 0), 2),
    };
}

test {
    _ = @import("int_test.zig");
}
