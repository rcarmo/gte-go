// sgemm_blocked_arm64.s — ARM64 NEON blocked NT tile kernel
//
// func sgemmNTTileFMA(iLen, jLen, kLen int, alpha float32,
//     a, b, c unsafe.Pointer, lda, ldb, ldc int)

#include "textflag.h"

// VFADD Vd.4S, Vn.4S, Vm.4S
#define VFADD(vm, vn, vd) WORD $(0x4e20d400 | ((vm)<<16) | ((vn)<<5) | (vd))

TEXT ·sgemmNTTileFMA(SB), NOSPLIT, $32-80
    MOVD    iLen+0(FP), R0         // i counter
    MOVD    jLen+8(FP), R1         // jLen constant
    MOVD    kLen+16(FP), R2        // kLen constant
    FMOVS   alpha+24(FP), F31      // alpha
    MOVD    a+32(FP), R3           // A base
    MOVD    lda+40(FP), R4         // lda
    MOVD    b+48(FP), R5           // B base
    MOVD    ldb+56(FP), R6         // ldb
    MOVD    c+64(FP), R7           // C base
    MOVD    ldc+72(FP), R8         // ldc

    LSL     $2, R4, R4             // lda_bytes
    LSL     $2, R6, R6             // ldb_bytes
    MOVD    R6, 24(RSP)            // save ldb_bytes
    LSL     $2, R8, R8             // ldc_bytes

    CBZ     R0, done

    MOVD    R3, R9                 // A row ptr
    MOVD    R7, R10                // C row ptr

i_loop:
    MOVD    R1, R11                // j counter
    MOVD    R5, R12                // B row ptr
    MOVD    R10, R13               // C element ptr

j_loop:
    CBZ     R11, next_i

    // Save j state
    MOVD    R11, 0(RSP)
    MOVD    R13, 8(RSP)
    MOVD    R12, 16(RSP)

    // k-loop: dot(A[i,:], B[j,:])
    MOVD    R9, R14                // A ptr
    MOVD    R12, R15               // B ptr
    MOVD    R2, R16                // k counter

    VEOR    V0.B16, V0.B16, V0.B16
    VEOR    V1.B16, V1.B16, V1.B16
    VEOR    V2.B16, V2.B16, V2.B16
    VEOR    V3.B16, V3.B16, V3.B16

    CMP     $16, R16
    BLT     k4

k16:
    VLD1.P  64(R14), [V4.S4, V5.S4, V6.S4, V7.S4]
    VLD1.P  64(R15), [V16.S4, V17.S4, V18.S4, V19.S4]
    VFMLA   V4.S4, V16.S4, V0.S4
    VFMLA   V5.S4, V17.S4, V1.S4
    VFMLA   V6.S4, V18.S4, V2.S4
    VFMLA   V7.S4, V19.S4, V3.S4
    SUB     $16, R16, R16
    CMP     $16, R16
    BGE     k16

k4:
    // Merge accumulators
    VFADD(1, 0, 0)                 // V0 += V1
    VFADD(3, 2, 2)                 // V2 += V3
    VFADD(2, 0, 0)                 // V0 += V2

    CMP     $4, R16
    BLT     k_reduce

k4_loop:
    VLD1.P  16(R14), [V4.S4]
    VLD1.P  16(R15), [V16.S4]
    VFMLA   V4.S4, V16.S4, V0.S4
    SUB     $4, R16, R16
    CMP     $4, R16
    BGE     k4_loop

k_reduce:
    // Horizontal sum V0.S4 → F0
    VMOV    V0.S[0], R17
    FMOVS   R17, F0
    VMOV    V0.S[1], R17
    FMOVS   R17, F4
    FADDS   F4, F0, F0
    VMOV    V0.S[2], R17
    FMOVS   R17, F4
    FADDS   F4, F0, F0
    VMOV    V0.S[3], R17
    FMOVS   R17, F4
    FADDS   F4, F0, F0

    // Scalar tail
    CBZ     R16, k_store

k_scalar:
    FMOVS   (R14), F4
    FMOVS   (R15), F5
    FMADDS  F4, F0, F5, F0        // F0 = F5*F4 + F0
    ADD     $4, R14
    ADD     $4, R15
    SUB     $1, R16, R16
    CBNZ    R16, k_scalar

k_store:
    // Restore j state
    MOVD    0(RSP), R11
    MOVD    8(RSP), R13
    MOVD    16(RSP), R12

    // C[i,j] += alpha * dot
    FMULS   F31, F0, F0
    FMOVS   (R13), F4
    FADDS   F0, F4, F4
    FMOVS   F4, (R13)

    // Advance j
    ADD     $4, R13
    MOVD    24(RSP), R6
    ADD     R6, R12
    SUB     $1, R11, R11
    B       j_loop

next_i:
    ADD     R4, R9                 // A next row
    ADD     R8, R10                // C next row
    SUB     $1, R0, R0
    CBNZ    R0, i_loop

done:
    RET
