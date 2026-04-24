# Benchmarks

## Cross-platform comparison

All benchmarks use `go test -bench=BenchmarkEmbed -benchtime=10s -count=3`.

### amd64 — Intel i7-12700, 6 P-cores, 4.7GHz

| Configuration | ms/embed | vs original | Allocs/op | B/op |
|---|---|---|---|---|
| Original (scalar, f64 math) | 67.4 | 1.0× | 170 | 15,598 |
| + gonum BLAS | 7.4 | 9.1× | 1,610 | 159,437 |
| + fused QKV + contiguous heads | 6.3 | 10.7× | 1,406 | 142,338 |
| **Pure Go + SIMD asm** | **6.4** | **10.5×** | 1,404 | 141,114 |
| OpenBLAS CGo (opt-in) | **5.5** | **12.3×** | 12 | 186 |

### arm64 — CIX P1 CD8160 (OrangePi 6 Plus), 12 cores, ~2GHz

| Configuration | ms/embed | vs gonum baseline | Allocs/op | B/op |
|---|---|---|---|---|
| gonum only (no SIMD) | 104 | 1.0× | 1,404 | 141,152 |
| + NEON GEBP (initial) | 64 | 1.6× | 12 | 209 |
| **+ edge tile fix** | **26** | **5.2×** | **12** | **194** |

### Parallel batch (8 texts)

| Platform | Mode | Total ms | Per-text ms |
|---|---|---|---|
| amd64 | Sequential | 52 | 6.5 |
| amd64 | Parallel (6 workers) | 56 | 7.0 |
| arm64 | Sequential | 215 | 26.9 |

### Load time

| Method | amd64 | arm64 |
|---|---|---|
| `Load()` | 0.20s | 0.25s |
| `LoadMmap()` | 0.01s | 0.01s |

## SVE status (arm64)

The CIX P1 CD8160 supports SVE and SVE2, but the default vector length is
128 bits (same as NEON). No performance benefit from SVE on this hardware.

## Architecture dispatch

| Path | amd64 | arm64 |
|---|---|---|
| NT (linear) | gonum (fast asm DotUnitary) | GEBP NEON micro-kernel |
| NN (attention) | AVX2+FMA tiled kernel | NEON tiled kernel |
| Attention Sdot | AVX2 FMA | NEON VFMLA |

### Why gonum wins NT on amd64

gonum's `f32.DotUnitary` has zero per-dot overhead — the function returns a
scalar that Go adds to C with a single instruction.  Despite not using FMA
(VMULPS+VADDPS), its deep pipelining with 4 accumulators saturates the
execution ports.  The 1,404 goroutine allocations cost ~70µs (1% of runtime).

### Why GEBP wins NT on arm64

gonum has no NEON assembly for NT — it falls back to scalar Go code (104ms).
Our GEBP NEON micro-kernel with edge-tile optimization brings it to 21ms.
