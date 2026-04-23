# Benchmarks

## Cross-platform comparison

All benchmarks use `go test -bench=BenchmarkEmbed -benchtime=10s -count=3`.
Pure Go builds with `CGO_ENABLED=0`.

### amd64 — Intel i7-12700, 6 P-cores, 4.7GHz

| Configuration | ms/embed | vs original | Allocs/op | B/op |
|---|---|---|---|---|
| Original (scalar, f64 math) | 67.4 | 1.0× | 170 | 15,598 |
| + gonum BLAS | 7.4 | 9.1× | 1,610 | 159,437 |
| + fused QKV + contiguous heads | 6.3 | 10.7× | 1,406 | 142,338 |
| **Pure Go + SIMD asm** | **6.4** | **10.5×** | **1,404** | **141,114** |
| OpenBLAS CGo (opt-in) | **5.5** | **12.3×** | **12** | **186** |

### arm64 — CIX P1 CD8160 (OrangePi 6 Plus), 12 cores, ~2GHz

| Configuration | ms/embed | vs gonum baseline | Allocs/op | B/op |
|---|---|---|---|---|
| gonum only (no SIMD) | 104 | 1.0× | 1,404 | 141,152 |
| **+ NEON GEBP + NN assembly** | **64** | **1.6×** | **12** | **209** |

### Parallel batch (8 texts, amd64)

| Mode | Total ms | Per-text ms | Speedup vs sequential |
|---|---|---|---|
| Sequential | 64 | 8.0 | 1.0× |
| Parallel (6 workers) | 56 | 7.0 | 1.1× |

Parallel batching provides modest throughput improvement for small texts.
The benefit increases with longer texts and larger batch sizes.

## SVE status (arm64)

The CIX P1 CD8160 supports SVE and SVE2, but the default vector length is
128 bits (same as NEON). No performance benefit from SVE on this hardware.

## Load time comparison

| Method | amd64 | arm64 |
|---|---|---|
| `Load()` | 0.20s | 0.25s |
| `LoadMmap()` | 0.01s | 0.01s |

`LoadMmap()` is recommended for OpenBLAS builds (same inference speed, 12×
faster startup). For pure Go builds on amd64, use `Load()` (gonum's tiled
reads cause page faults on mmap'd memory).
