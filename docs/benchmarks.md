# Benchmarks

## Final results

| Platform | CPU | Build | ms/embed | vs baseline | Allocs | Headroom¹ |
|---|---|---|---|---|---|---|
| **amd64** | i7-12700, 4.7GHz | Pure Go + SIMD | **6.4** | **10.5×** | 1,404 | 1.6× |
| **amd64** | i7-12700, 4.7GHz | OpenBLAS CGo | **5.5** | **12.3×** | 12 | 1.4× |
| **arm64** | CIX P1 CD8160, 2GHz | Pure Go + NEON | **20** | **5.2×** | 12 | 1.4× |

¹ Ratio of measured time to theoretical compute+bandwidth floor.

## Optimization progression

### amd64

| Stage | ms/embed | vs original | Change |
|---|---|---|---|
| Original | 67.4 | 1.0× | Scalar loops, math.Tanh via float64 |
| + float32 fast math | 63.4 | 1.06× | Padé/Newton approximations |
| + gonum BLAS | 7.4 | 9.1× | Cache-blocked matrix multiply |
| + adaptive attention | 7.3 | 9.2× | No goroutines for short sequences |
| + fused QKV | 6.6 | 10.2× | 3 Sgemm → 1 |
| + contiguous heads | 6.3 | 10.7× | [heads, seq, dim] layout |
| + OpenBLAS CGo | 5.5 | 12.3× | cblas_sgemm via CGo |
| **Final (pure Go)** | **6.4** | **10.5×** | gonum NT + AVX2 NN assembly |

### arm64

| Stage | ms/embed | vs gonum | Change |
|---|---|---|---|
| gonum only | 104 | 1.0× | No NEON for NT path |
| + GEBP NEON micro-kernel | 64 | 1.6× | 4×16 VFMLA tile |
| + edge tile fix | 26 | 4.0× | Route partial mr through assembly |
| + NEON pack transpose | 21 | 5.0× | 4×4 VTRN/VZIP vectorized pack |
| **+ fused residual+layerNorm** | **20** | **5.2×** | Single-pass add+normalize |

### arm64 profile breakdown (20ms)

| Component | % | ms | Status |
|---|---|---|---|
| GEBP micro-kernel (NEON) | 44% | 8.7 | Near theoretical (9ms floor) |
| Pack transpose (NEON) | 32% | 6.4 | Bandwidth-limited (~5ms floor) |
| Model load (binary.Read) | 10% | 2.0 | Not in hot path |
| gelu (fastTanh) | 3% | 0.6 | Scalar, diminishing returns |
| Attention + softmax | 2% | 0.4 | Already optimized |
| Other | 9% | 1.9 | Runtime, GC, misc |

## Architecture dispatch

| Path | amd64 | arm64 |
|---|---|---|
| NT (all linear) | gonum (asm DotUnitary) | GEBP NEON micro-kernel |
| NN (attention) | AVX2+FMA tiled kernel | NEON tiled kernel |
| Attention Sdot | AVX2 FMA | NEON VFMLA |
| Pack | — | NEON 4×4 transpose |

## SVE (arm64)

CIX P1 CD8160 supports SVE2 but vector length is 128 bits (same as NEON). No benefit.

## Load methods

| Build | `Load()` | `LoadMmap()` | Recommendation |
|---|---|---|---|
| Pure Go (amd64) | 6.4 ms, 0.20s | slower ⚠️ | **Use `Load()`** |
| Pure Go (arm64) | 20 ms, 0.25s | 20 ms, 0.01s | Either works |
| OpenBLAS CGo | 5.5 ms, 0.20s | 5.5 ms, 0.01s | **Use `LoadMmap()`** |
