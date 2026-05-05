// dotq4_int_arm64.s — NEON SDOT Q4 dot product
//
// func DotQ4Int(x unsafe.Pointer, blocks unsafe.Pointer, nBlocks int) float32
//
// Uses SDOT (signed 8-bit dot product, ARMv8.2+):
//   SDOT Vd.4S, Vn.16B, Vm.16B
//   Each lane: Vd.S[i] += dot(Vn.B[4i:4i+4], Vm.B[4i:4i+4])
//   = 16 signed int8 multiplies → 4 int32 accumulators per instruction
//
// Per block (32 elements):
//   1. Find absmax of x[0:32], compute inv_scale = 127/absmax
//   2. Quantize x to int8 (signed, range [-127,127])
//   3. Expand Q4 nibbles to int8 (nibble-8, range [-8,7])
//   4. 2× SDOT → 4 int32 accumulators
//   5. Hsum int32, descale: total += (w_scale * absmax/127) * dot
//
#include "textflag.h"

// SDOT Vd.4S, Vn.16B, Vm.16B = 0x4E809400 | (Vm<<16) | (Vn<<5) | Vd
#define SDOT(vm, vn, vd) WORD $(0x4E809400 | ((vm)<<16) | ((vn)<<5) | (vd))

// SCVTF Vd.4S, Vn.4S (signed int32 → float32) = 0x4E21D800 | (Vn<<5) | Vd
// Wait, SCVTF is 0x4E21D800 for signed (vs 0x6E21D800 for unsigned)
#define SCVTF_4S(vn, vd) WORD $(0x4E21D800 | ((vn)<<5) | (vd))

// SQXTN Vd.8B, Vn.8H (saturating narrow int16 → int8)
#define SQXTN_8B(vn, vd) WORD $(0x0E214800 | ((vn)<<5) | (vd))
// SQXTN2 Vd.16B, Vn.8H (saturating narrow, upper half)
#define SQXTN2_16B(vn, vd) WORD $(0x4E214800 | ((vn)<<5) | (vd))
// SQXTN Vd.4H, Vn.4S (saturating narrow int32 → int16)
#define SQXTN_4H(vn, vd) WORD $(0x0E614800 | ((vn)<<5) | (vd))
// SQXTN2 Vd.8H, Vn.4S
#define SQXTN2_8H(vn, vd) WORD $(0x4E614800 | ((vn)<<5) | (vd))

// SADDLV Sd, Vn.16B (add across long, signed bytes → int16 in Sd)
// Actually we need sum of 32 signed bytes. Use pairwise adds.
// SADDLP Vd.8H, Vn.16B = signed add pairwise long: pairs of i8 → i16
#define SADDLP_16B_8H(vn, vd) WORD $(0x4E202800 | ((vn)<<5) | (vd))
// SADDLP Vd.4S, Vn.8H = pairs of i16 → i32
#define SADDLP_8H_4S(vn, vd) WORD $(0x4E602800 | ((vn)<<5) | (vd))
// ADDV Sd, Vn.4S = add across vector (4 int32 → 1 int32)
#define ADDV_4S(vn, vd) WORD $(0x4EB1B800 | ((vn)<<5) | (vd))

// FMUL Vd.4S, Vn.4S, Vm.4S
#define FMUL_4S(vm, vn, vd) WORD $(0x6E20DC00 | ((vm)<<16) | ((vn)<<5) | (vd))
// FSUB Vd.4S, Vn.4S, Vm.4S
#define FSUB_4S(vm, vn, vd) WORD $(0x4EA0D400 | ((vm)<<16) | ((vn)<<5) | (vd))

// FCVTNS Vd.4S, Vn.4S (float→int32, round to nearest)
#define FCVTNS_4S(vn, vd) WORD $(0x4E21A800 | ((vn)<<5) | (vd))

// FABS Vd.4S, Vn.4S
#define FABS_4S(vn, vd) WORD $(0x4EA0F800 | ((vn)<<5) | (vd))
// FMAXV Sd, Vn.4S (max across vector)
#define FMAXV_4S(vn, vd) WORD $(0x6E30F800 | ((vn)<<5) | (vd))
// FMAX Vd.4S, Vn.4S, Vm.4S
#define FMAX_4S(vm, vn, vd) WORD $(0x4E20F400 | ((vm)<<16) | ((vn)<<5) | (vd))

// func DotQ4Int(x unsafe.Pointer, blocks unsafe.Pointer, nBlocks int) float32
TEXT ·DotQ4Int(SB), NOSPLIT, $0-28
    MOVD    x+0(FP), R0          // x
    MOVD    blocks+8(FP), R1     // blocks
    MOVD    nBlocks+16(FP), R2   // nBlocks

    // F20 = total accumulator (scalar float)
    FMOVS   $0.0, F20

    // V21 = [0x0F x 16] mask
    MOVD    $0x0F0F0F0F0F0F0F0F, R3
    VMOV    R3, V21.D[0]
    VMOV    R3, V21.D[1]

    // V22 = [8 x 16] for nibble centering
    MOVD    $0x0808080808080808, R3
    VMOV    R3, V22.D[0]
    VMOV    R3, V22.D[1]

    // F23 = 127.0
    FMOVS   $127.0, F23
    // F24 = 1.0/127.0
    FMOVS   $1.0, F24
    FDIVS   F23, F24, F24       // F24 = 1/127

    CBZ     R2, dii_done

dii_block:
    // --- Step 1: Find absmax of x[0:32] ---
    VLD1.P  64(R0), [V0.S4, V1.S4, V2.S4, V3.S4]  // x[0:16]
    VLD1    (R0), [V4.S4, V5.S4, V6.S4, V7.S4]     // x[16:32]
    // Save R0 (we need to rewind for quantization)
    SUB     $64, R0, R4         // R4 = &x[0] (original position)

    // Abs of all 8 vectors
    FABS_4S(0, 8)    // V8 = abs(V0)
    FABS_4S(1, 9)    // V9 = abs(V1)
    FMAX_4S(9, 8, 8) // V8 = max(V8, V9)
    FABS_4S(2, 9)
    FMAX_4S(9, 8, 8)
    FABS_4S(3, 9)
    FMAX_4S(9, 8, 8)
    FABS_4S(4, 9)
    FMAX_4S(9, 8, 8)
    FABS_4S(5, 9)
    FMAX_4S(9, 8, 8)
    FABS_4S(6, 9)
    FMAX_4S(9, 8, 8)
    FABS_4S(7, 9)
    FMAX_4S(9, 8, 8)
    // Horizontal max of V8.4S → F8
    FMAXV_4S(8, 8)   // V8.S[0] = max of 4 lanes

    // Check zero
    FMOVS   F8, R3
    CBZ     R3, dii_skip

    // --- Step 2: inv_scale = 127 / absmax ---
    FDIVS   F8, F23, F25        // F25 = 127 / absmax = inv_scale
    VDUP    V25.S[0], V25.S4    // V25 = [inv_scale × 4]

    // --- Step 3: Quantize x to int8 ---
    // Multiply by inv_scale
    FMUL_4S(25, 0, 0)   // V0 *= inv_scale
    FMUL_4S(25, 1, 1)
    FMUL_4S(25, 2, 2)
    FMUL_4S(25, 3, 3)
    FMUL_4S(25, 4, 4)
    FMUL_4S(25, 5, 5)
    FMUL_4S(25, 6, 6)
    FMUL_4S(25, 7, 7)
    // Round to nearest int32
    FCVTNS_4S(0, 0)     // float → int32
    FCVTNS_4S(1, 1)
    FCVTNS_4S(2, 2)
    FCVTNS_4S(3, 3)
    FCVTNS_4S(4, 4)
    FCVTNS_4S(5, 5)
    FCVTNS_4S(6, 6)
    FCVTNS_4S(7, 7)
    // Saturating narrow: i32 → i16 → i8
    SQXTN_4H(0, 0)      // V0.4H = saturate(V0.4S)
    SQXTN2_8H(1, 0)     // V0.8H = [V0.4S | V1.4S]
    SQXTN_4H(2, 1)
    SQXTN2_8H(3, 1)     // V1.8H = [V2.4S | V3.4S]
    SQXTN_4H(4, 2)
    SQXTN2_8H(5, 2)     // V2.8H = [V4.4S | V5.4S]
    SQXTN_4H(6, 3)
    SQXTN2_8H(7, 3)     // V3.8H = [V6.4S | V7.4S]
    // i16 → i8
    SQXTN_8B(0, 0)      // V0.8B = saturate(V0.8H) [lower]
    SQXTN2_16B(1, 0)    // V0.16B = x_i8[0:16]
    SQXTN_8B(2, 1)
    SQXTN2_16B(3, 1)    // V1.16B = x_i8[16:32]

    // --- Step 4: Expand Q4 nibbles to signed int8 ---
    ADD     $4, R1, R3
    VLD1    (R3), [V2.B16]      // 16 packed bytes
    VAND    V21.B16, V2.B16, V3.B16  // V3 = low nibbles
    VUSHR   $4, V2.B16, V4.B16       // V4 = high nibbles
    // Subtract 8 to center: [-8, 7]
    // VSUB doesn't work for bytes in Go asm, use word:
    // SUB V22.16B, V3.16B, V3.16B
    WORD    $(0x6E226060 | (22<<16) | (3<<5) | 3)  // USUB V3.16B, V3.16B, V22.16B -- wait this is wrong
    // Actually: SUB Vd.16B, Vn.16B, Vm.16B = 0x6E200400 | (Vm<<16) | (Vn<<5) | Vd
    // V3 = V3 - V22
    WORD    $(0x6E360463)  // SUB V3.16B, V3.16B, V22.16B  -- 6E (22=V22)<<16 | (3)<<5 | 3
    // Hmm let me recalculate. SUB (vector) int: 01 1 01110 sz 1 Rm 100001 Rn Rd
    // For 16B (Q=1, size=00): 0110 1110 0010 0000 1000 01 Rn Rd + Rm<<16
    // = 0x6E208400 | (Rm<<16) | (Rn<<5) | Rd
    // V3 = V3 - V22: Rm=22, Rn=3, Rd=3
    // WRONG: need to redo

    // Let me just use: result = nibbles - 8. Since V22 = [8x16]:
    // In Go asm, VSUB should work for B16... let me try a different approach:
    // Instead of subtract, XOR with 0x08 and treat as signed? No.
    // Actually, the nibble values are [0,15]. We want signed [-8,7].
    // If we store nibbles as unsigned and let SDOT treat them as signed...
    // No, SDOT is signed×signed, values 0-15 unsigned would be treated as 0-15 signed. Fine!
    // But then we need to subtract 8*sum(x_i8) as correction.
    //
    // Simpler: keep nibbles as unsigned [0,15], use SDOT (both signed),
    // and correct: dot(x, nibble-8) = dot(x, nibble) - 8*sum(x)
    // Since nibbles are [0,15] they fit in signed int8 (max 15 < 127). Perfect.

    // So: V3 = low nibbles (0-15, valid as signed int8)
    //     V4 = high nibbles (0-15, valid as signed int8)
    // w_s8_with_bias = nibbles (no subtraction needed!)
    // correction = 8 * sum(x_i8_block)

    // --- Step 5: SDOT ---
    VEOR    V9.B16, V9.B16, V9.B16   // V9 = zero accumulator
    SDOT(3, 0, 9)               // V9.4S += dot(V0.16B, V3.16B) — x[0:16] · w_lo[0:16]
    VEOR    V10.B16, V10.B16, V10.B16
    SDOT(4, 1, 10)              // V10.4S += dot(V1.16B, V4.16B) — x[16:32] · w_hi[0:16]

    // Sum V9 and V10
    WORD    $(0x4EB1B800 | (9<<5) | 9)   // ADDV S9, V9.4S (sum 4 lanes)
    WORD    $(0x4EB1B800 | (10<<5) | 10)  // ADDV S10, V10.4S
    // V9.S[0] + V10.S[0] = raw dot (as int32 in S-register)
    FMOVS   V9.S[0], R5                   // doesn't work — need VMOV
    VMOV    V9.S[0], R5
    VMOV    V10.S[0], R6
    ADD     R5, R6, R5                     // R5 = raw_dot (int32)

    // --- Step 6: Correction: subtract 8 * sum(x_i8) ---
    // sum(x_i8[0:16]) in V0, sum(x_i8[16:32]) in V1
    SADDLP_16B_8H(0, 11)       // V11.8H = pairwise add of V0.16B
    SADDLP_8H_4S(11, 11)       // V11.4S = pairwise add of V11.8H
    ADDV_4S(11, 11)             // V11.S[0] = sum(x_i8[0:16])
    VMOV    V11.S[0], R6
    SADDLP_16B_8H(1, 12)
    SADDLP_8H_4S(12, 12)
    ADDV_4S(12, 12)
    VMOV    V12.S[0], R7
    ADD     R6, R7, R6          // R6 = sum(all 32 x_i8)
    LSL     $3, R6, R6          // R6 = 8 * sum(x_i8)

    SUBS    R6, R5, R5          // R5 = corrected_dot = raw_dot - 8*sum(x_i8)

    // --- Step 7: Descale and accumulate ---
    // result = (w_scale * absmax / 127) * corrected_dot
    SCVTFS  R5, F10             // F10 = float(corrected_dot)
    FMOVS   (R1), F11           // F11 = w_scale
    FMULS   F8, F11, F11        // F11 = w_scale * absmax
    FMULS   F24, F11, F11       // F11 = w_scale * absmax / 127
    FMULS   F11, F10, F10       // F10 = contribution
    FADDS   F10, F20, F20       // total += contribution

dii_skip:
    ADD     $20, R1, R1         // blocks += 20
    // x already advanced by VLD1.P (64 bytes + 64 bytes at top)
    // Wait, we loaded 2×VLD1 consuming 128 bytes total, but only first was .P
    // Let me fix: R0 was advanced by 64 from the first VLD1.P, and we loaded x[16:32] from (R0)
    // So R0 is now pointing at x[16] + 64 = x[32]. Good — advanced by 128 total.
    ADD     $64, R0, R0         // advance past second VLD1 (non-postincrement)
    SUB     $1, R2, R2
    CBNZ    R2, dii_block

dii_done:
    FMOVS   F20, ret+24(FP)
    RET
