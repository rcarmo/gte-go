# Optimization Report

This document details the systematic optimization of `gte-go` from the original
67.4ms per embedding down to 5.5ms (OpenBLAS) / 6.4ms (pure Go) on amd64, and
64ms on arm64.

## Approach

Three phases, each building on the previous:

1. **BLAS and fast math** — replace scalar loops with optimized matrix multiply
2. **Structural optimizations** — fuse operations, improve memory layout
3. **System-level** — SIMD assembly, mmap, zero-alloc tokenizer

## Phase 1 — BLAS and fast math (67.4 → 7.3ms)

### 1a – float32 fast math (67.4 → 63.4ms, 1.06×)

Replaced `math.Tanh`, `math.Exp`, and `math.Sqrt` (all float64 round-trips)
with float32 polynomial/Padé approximations in `gte/fastmath.go`:

- `fastTanh`: degree-7 Padé approximant, clamped at ±5
- `fastExp`: range-reduced polynomial with bit-manipulation reconstruction
- `fastInvSqrt`: Quake III bit trick + 2 Newton-Raphson steps

Modest 6% gain because `linear()` dominates.

### 1b – gonum BLAS (63.4 → 7.4ms, 9.1×)

Replaced the hand-rolled `linear()` dot-product loops with `blas32.Sgemm` from
gonum. This was the single largest improvement. Even gonum's pure-Go BLAS with
cache blocking massively outperforms scalar code.

### 1c – adaptive attention (7.4 → 7.3ms, 9.2×)

Short sequences (< 32 tokens) use a scalar attention path without goroutine
overhead. Longer sequences use BLAS for Q·K^T and attn·V.

## Phase 2 — Structural optimizations (7.3 → 6.3ms)

### 2a – fused QKV projection (7.3 → 6.6ms, 10.2×)

Concatenated Q/K/V weight matrices into a single `[3*hidden, hidden]` array at
load time. One Sgemm call replaces three.

### 2b – contiguous head layout (6.6 → 6.3ms, 10.7×)

Changed the QKV split to write directly into `[numHeads, seqLen, headDim]`
buffers. Eliminates strided access in attention inner loops.

### 2c – sigmoid GELU (abandoned)

Tested but the approximation error exceeded the model's cosine similarity
tolerance by 0.006.

## Phase 3 — System-level optimizations (6.3 → 5.5ms)

### 3a – OpenBLAS via CGo (6.3 → 5.5ms, 12.3×)

Direct CGo wrapper for `cblas_sgemm`, bypassing gonum entirely. Eliminated
99% of per-embedding allocations (gonum's `sgemmParallel` spawns goroutines).
OpenBLAS provides AVX2 SIMD acceleration.

### 3b – zero-alloc tokenizer

Refactored tokenizer to reuse pre-allocated buffers. Reduced remaining allocs
from 14 → 12, 1.4KB → 186B.

### 3c – mmap model loading

`LoadMmap()` maps the `.gtemodel` file directly. Weight slices point into the
mapped region — no copy into Go heap. 12× faster startup, ~127MB less resident
memory.

## Phase 4 — SIMD assembly

### AVX2+FMA kernels (amd64)

- `Sdot`: 16-wide dot product with FMA, 4-way unrolled
- `Saxpy`: 16-wide y += alpha*x with FMA
- `SgemmNN`: 32-wide C tiling, register-blocked k loop
- `SgemmNT`: 2j-at-a-time dot products, shared A loads
- GEBP 6×16 micro-kernel for large-m NT

### NEON kernels (arm64)

- `Sdot`, `Saxpy`: 8-wide with VFMLA
- `SgemmNN`: 16-wide C tiling
- `SgemmNT`: blocked tile with VFMLA, prefetch hints
- GEBP 4×16 micro-kernel for NT dispatch

### Architecture-specific dispatch

The NT matmul (used by all `linear()` calls) is the dominant bottleneck.
The optimal strategy differs per architecture:

| Path | amd64 | arm64 |
|---|---|---|
| NT, small m | gonum (its asm DotUnitary + cache blocking) | GEBP + NEON |
| NT, large m | GEBP + AVX2 | GEBP + NEON |
| NN | AVX2+FMA tiled kernel | NEON tiled kernel |

**Why gonum wins NT on amd64**: gonum's `f32.DotUnitary` uses hand-written x86
assembly with deep pipelining. Although it doesn't use FMA (VMULPS+VADDPS
instead of VFMADD231PS), its zero-overhead per-dot-product design beats our
FMA tile kernel which has horizontal-reduce overhead per output element.

**Why GEBP wins NT on arm64**: gonum has no NEON assembly for NT — it falls
back to scalar Go code. Our NEON GEBP micro-kernel is 1.6× faster.

## Key insight: the horizontal reduce problem

For NT GEMM, each output element C[i,j] = dot(A[i,:], B[j,:]) requires a
horizontal sum at the end of the dot product. On x86, this is 5 instructions
(VEXTRACTF128 + VADDPS + 2×VHADDPS). For 8064 output elements per sgemm call,
that's 40K overhead instructions — 10% of the total FMA work.

OpenBLAS avoids this by using the GEBP approach with B-packing, which converts
the NT dot products into NN-style fused multiply-accumulate without horizontal
reductions. This is the main reason OpenBLAS achieves 5.5ms vs our 6.4ms.
