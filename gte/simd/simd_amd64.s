// simd_amd64.s — AVX2/FMA SIMD kernels for Go (pure Go build, no CGo)
//
// func sdot(x, y []float32) float32
//   - Dot product of two float32 slices using AVX2+FMA.
//
// func saxpy(alpha float32, x []float32, y []float32)
//   - y[i] += alpha * x[i], SIMD-accelerated.

#include "textflag.h"

// func sdot(x, y []float32) float32
TEXT ·Sdot(SB), NOSPLIT, $0-52
    MOVQ    x_base+0(FP), SI
    MOVQ    x_len+8(FP), CX
    MOVQ    y_base+24(FP), DI

    VXORPS  Y0, Y0, Y0
    VXORPS  Y1, Y1, Y1

    CMPQ    CX, $16
    JL      sdot_tail8

sdot_loop16:
    VMOVUPS (SI), Y2
    VMOVUPS 32(SI), Y3
    VFMADD231PS (DI), Y2, Y0
    VFMADD231PS 32(DI), Y3, Y1
    ADDQ    $64, SI
    ADDQ    $64, DI
    SUBQ    $16, CX
    CMPQ    CX, $16
    JGE     sdot_loop16

sdot_tail8:
    CMPQ    CX, $8
    JL      sdot_tail4
    VMOVUPS (SI), Y2
    VFMADD231PS (DI), Y2, Y0
    ADDQ    $32, SI
    ADDQ    $32, DI
    SUBQ    $8, CX

sdot_tail4:
    VADDPS  Y1, Y0, Y0
    CMPQ    CX, $4
    JL      sdot_reduce
    VMOVUPS (SI), X2
    VFMADD231PS (DI), X2, X0
    ADDQ    $16, SI
    ADDQ    $16, DI
    SUBQ    $4, CX

sdot_reduce:
    VEXTRACTF128 $1, Y0, X1
    VADDPS  X1, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0

    CMPQ    CX, $0
    JE      sdot_done

sdot_scalar:
    VMOVSS  (SI), X1
    VMOVSS  (DI), X2
    VFMADD231SS X2, X1, X0
    ADDQ    $4, SI
    ADDQ    $4, DI
    SUBQ    $1, CX
    JNZ     sdot_scalar

sdot_done:
    VMOVSS  X0, ret+48(FP)
    VZEROUPPER
    RET

// func saxpy(alpha float32, x []float32, y []float32)
// y[i] += alpha * x[i]
TEXT ·Saxpy(SB), NOSPLIT, $0-56
    MOVSS       alpha+0(FP), X8
    VBROADCASTSS X8, Y8             // Y8 = [alpha, alpha, ...]
    MOVQ    x_base+8(FP), SI       // SI = &x[0]
    MOVQ    x_len+16(FP), CX       // CX = len(x)
    MOVQ    y_base+32(FP), DI      // DI = &y[0]

    CMPQ    CX, $16
    JL      saxpy_tail8

saxpy_loop16:
    VMOVUPS (DI), Y0
    VMOVUPS 32(DI), Y1
    VMOVUPS (SI), Y2
    VMOVUPS 32(SI), Y3
    VFMADD231PS Y8, Y2, Y0         // y[i:i+8] += alpha * x[i:i+8]
    VFMADD231PS Y8, Y3, Y1
    VMOVUPS Y0, (DI)
    VMOVUPS Y1, 32(DI)
    ADDQ    $64, SI
    ADDQ    $64, DI
    SUBQ    $16, CX
    CMPQ    CX, $16
    JGE     saxpy_loop16

saxpy_tail8:
    CMPQ    CX, $8
    JL      saxpy_tail4
    VMOVUPS (DI), Y0
    VMOVUPS (SI), Y2
    VFMADD231PS Y8, Y2, Y0
    VMOVUPS Y0, (DI)
    ADDQ    $32, SI
    ADDQ    $32, DI
    SUBQ    $8, CX

saxpy_tail4:
    CMPQ    CX, $4
    JL      saxpy_scalar
    VMOVUPS (DI), X0
    VMOVUPS (SI), X2
    VFMADD231PS X8, X2, X0
    VMOVUPS X0, (DI)
    ADDQ    $16, SI
    ADDQ    $16, DI
    SUBQ    $4, CX

saxpy_scalar:
    CMPQ    CX, $0
    JE      saxpy_done

saxpy_scalar_loop:
    VMOVSS  (DI), X0
    VMOVSS  (SI), X2
    VFMADD231SS X8, X2, X0
    VMOVSS  X0, (DI)
    ADDQ    $4, SI
    ADDQ    $4, DI
    SUBQ    $1, CX
    JNZ     saxpy_scalar_loop

saxpy_done:
    VZEROUPPER
    RET
