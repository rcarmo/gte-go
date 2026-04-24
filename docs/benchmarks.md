# Benchmarks

## Headline numbers

| Platform | ms/embed | Allocs | B/op | GC pressure @ 100 qps |
|---|---|---|---|---|
| **amd64** i7-12700 | **10** | **1** | **7** | **~700 B/s** |
| **arm64** CIX P1 CD8160 | **20** | **1** | **11** | **~1.1 KB/s** |
| amd64 OpenBLAS CGo | 5.5 | 1 | 7 | ~700 B/s |
| arm64 OpenBLAS CGo | 14 | 1 | 11 | ~1.1 KB/s |

## Latency jitter (5000 embeddings from Northwind DB)

708 unique text strings from [northwind.sql](../assets/northwind.sql), cycling through 5000 calls.

### Discrete (individual embeds)

| | amd64 | arm64 |
|---|---|---|
| **p50** | **12.9ms** | **20.4ms** |
| p95 | 25.1ms | 33.6ms |
| p99 | 49.2ms | 63.2ms |
| max | 155ms | 451ms |
| σ | 8.8ms | 22.9ms |

### Batch-100 (per-embed average within 100-row batches)

| | amd64 | arm64 |
|---|---|---|
| **p50** | **15.1ms** | **21.5ms** |
| p95 | 19.0ms | 35.7ms |
| p99 | 20.8ms | 36.7ms |
| σ | **2.4ms** | **4.7ms** |

Batching reduces variance 3–5× by amortizing Go runtime background work across 100 embeds. The remaining jitter is from Go GC/scavenger, not from our code (1 alloc of 7–11 bytes per embed).

### Reproduce

```bash
go build -o jitter ./cmd/jitter/
./jitter -model gte-small.gtemodel -texts assets/northwind_texts.txt -n 5000 -out discrete.csv
./jitter -model gte-small.gtemodel -texts assets/northwind_texts.txt -n 5000 -batch 100 -out batch100.csv
```

Raw data: [assets/](../assets/)

## Optimization progression

### amd64 — Intel i7-12700, 4.7GHz

| Stage | ms/embed | vs original | Allocs | Approach |
|---|---|---|---|---|
| Original | 67.4 | 1.0× | 170 | Scalar loops, math via float64 |
| + float32 fast math | 63.4 | 1.06× | 170 | Padé/Newton approximations |
| + gonum BLAS | 7.4 | 9.1× | 1,610 | Cache-blocked matmul |
| + fused QKV + heads | 6.3 | 10.7× | 1,406 | Structural |
| + OpenBLAS CGo | 5.5 | 12.3× | 12 | cblas_sgemm |
| **VGATHERDPS (pure Go)** | **10** | **6.7×** | **1** | AVX2 gather, zero pack |

### arm64 — CIX P1 CD8160, ~2GHz

| Stage | ms/embed | vs gonum | Allocs | Approach |
|---|---|---|---|---|
| gonum only | 104 | 1.0× | 1,404 | No NEON for NT |
| + GEBP NEON | 64 | 1.6× | 12 | 4×16 micro-kernel |
| + edge tile fix | 26 | 4.0× | 12 | Route partial tiles through asm |
| + NEON pack | 21 | 5.0× | 12 | 4×4 transpose |
| **+ zero-alloc tokenizer** | **20** | **5.2×** | **1** | unsafe.String, reusable buffers |
| OpenBLAS CGo | 14 | 7.4× | 1 | System BLAS |

### arm64 profile breakdown (20ms)

| Component | % | ms |
|---|---|---|
| GEBP micro-kernel (NEON) | 44% | 8.7 |
| Pack transpose (NEON) | 32% | 6.4 |
| gelu + layerNorm + attention | 8% | 1.6 |
| Other (Go runtime, fused ops) | 16% | 3.3 |

## GC pressure comparison

| Implementation | Allocs/embed | Bytes/embed | GC @ 100 qps | Goroutine churn |
|---|---|---|---|---|
| gonum BLAS | 1,404 | 141 KB | 13.4 MB/s | 140K/s |
| Previous (12 allocs) | 12 | 187 B | 18 KB/s | 0 |
| **Current (1 alloc)** | **1** | **7 B** | **~700 B/s** | **0** |

## Architecture dispatch

| Path | amd64 | arm64 |
|---|---|---|
| NT (linear) | VGATHERDPS 6×8 | GEBP NEON 4×16 |
| NN (attention) | AVX2+FMA tiled | NEON tiled |
| Attention Sdot | AVX2 FMA | NEON VFMLA |
| Pack | — | NEON 4×4 transpose |

## Load methods

| Build | `Load()` | `LoadMmap()` |
|---|---|---|
| Pure Go (amd64) | 10ms, 0.20s load | slower ⚠️ |
| Pure Go (arm64) | 20ms, 0.25s load | 20ms, 0.01s load |
| OpenBLAS CGo | 5.5ms/14ms, 0.20s | same, 0.01s load |

## SVE (arm64)

CIX P1 supports SVE2 but vector length = 128 bits (same as NEON). No benefit.
