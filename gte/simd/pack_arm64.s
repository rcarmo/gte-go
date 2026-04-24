// pack_arm64.s — NEON-accelerated B tile packing for GEBP
//
// func packBNTNeon(b0..b15 uintptr, k int, bp uintptr)
//
// Transposes 16 B rows into column-panel format: bp[p*16+d] = B[d][p]
// Processes 4 columns (p) at a time using VLD1 + VST1 interleaved stores.
//
// Args: 16 row pointers (128 bytes) + k (8 bytes) + bp (8 bytes) = 144 bytes

#include "textflag.h"

TEXT ·packBNTNeon(SB), NOSPLIT, $0-144
    // Load 16 B row pointers into R0-R15
    MOVD    b0+0(FP), R0
    MOVD    b1+8(FP), R1
    MOVD    b2+16(FP), R2
    MOVD    b3+24(FP), R3
    MOVD    b4+32(FP), R4
    MOVD    b5+40(FP), R5
    MOVD    b6+48(FP), R6
    MOVD    b7+56(FP), R7
    MOVD    b8+64(FP), R8
    MOVD    b9+72(FP), R9
    MOVD    b10+80(FP), R10
    MOVD    b11+88(FP), R11
    MOVD    b12+96(FP), R12
    MOVD    b13+104(FP), R13
    MOVD    b14+112(FP), R14
    MOVD    b15+120(FP), R15
    MOVD    k+128(FP), R16
    MOVD    bp+136(FP), R17

    // Process 1 column at a time: load 1 float from each of 16 rows,
    // store as 16 contiguous floats in bp.
    // This is simple and correct; the key speedup is avoiding Go overhead.
    CBZ     R16, pack_done

pack_loop:
    // Load 1 float32 from each of 16 B rows
    FMOVS   (R0), F0
    FMOVS   (R1), F1
    FMOVS   (R2), F2
    FMOVS   (R3), F3
    // Pack first 4 into V0
    VMOV    V0.S[0], V16.S[0]
    VMOV    V1.S[0], V16.S[1]
    VMOV    V2.S[0], V16.S[2]
    VMOV    V3.S[0], V16.S[3]

    FMOVS   (R4), F0
    FMOVS   (R5), F1
    FMOVS   (R6), F2
    FMOVS   (R7), F3
    VMOV    V0.S[0], V17.S[0]
    VMOV    V1.S[0], V17.S[1]
    VMOV    V2.S[0], V17.S[2]
    VMOV    V3.S[0], V17.S[3]

    FMOVS   (R8), F0
    FMOVS   (R9), F1
    FMOVS   (R10), F2
    FMOVS   (R11), F3
    VMOV    V0.S[0], V18.S[0]
    VMOV    V1.S[0], V18.S[1]
    VMOV    V2.S[0], V18.S[2]
    VMOV    V3.S[0], V18.S[3]

    FMOVS   (R12), F0
    FMOVS   (R13), F1
    FMOVS   (R14), F2
    FMOVS   (R15), F3
    VMOV    V0.S[0], V19.S[0]
    VMOV    V1.S[0], V19.S[1]
    VMOV    V2.S[0], V19.S[2]
    VMOV    V3.S[0], V19.S[3]

    // Store 16 floats contiguously
    VST1.P  [V16.S4, V17.S4, V18.S4, V19.S4], 64(R17)

    // Advance all row pointers by 4 bytes
    ADD     $4, R0
    ADD     $4, R1
    ADD     $4, R2
    ADD     $4, R3
    ADD     $4, R4
    ADD     $4, R5
    ADD     $4, R6
    ADD     $4, R7
    ADD     $4, R8
    ADD     $4, R9
    ADD     $4, R10
    ADD     $4, R11
    ADD     $4, R12
    ADD     $4, R13
    ADD     $4, R14
    ADD     $4, R15

    SUB     $1, R16, R16
    CBNZ    R16, pack_loop

pack_done:
    RET
