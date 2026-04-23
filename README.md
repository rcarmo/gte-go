# GTE-Small in Go

A pure Go implementation of the [GTE-small](https://huggingface.co/thenlper/gte-small) text embedding model. Produces 384-dimensional, L2-normalized embeddings suitable for similarity search and clustering, directly ported from [@antirez's C implementation](https://github.com/antirez/gte-pure-C).

The default build is **pure Go with SIMD assembly** — a single static binary with no C dependencies:

| Platform | Default (pure Go) | OpenBLAS CGo (opt-in) |
|---|---|---|
| **amd64** (i7-12700) | **10 ms/embed** (6.7×) | **5.5 ms** (12.3×) |
| **arm64** (CIX P1 CD8160) | **20 ms/embed** (5.2×¹) | — |

¹ vs gonum-only baseline on same hardware.  See [Optimization Report](#optimization-report) for full details.

## Quick Start

```bash
# Download Hugging Face weights and convert to .gtemodel
pip install safetensors requests numpy
python convert_model.py models/gte-small gte-small.gtemodel

# Build and run (pure Go, no C dependencies)
make run-go

# Or with OpenBLAS for maximum throughput (requires gcc + libopenblas-dev)
CGO_ENABLED=1 make run-go
```

Sample output:

```
Cosine similarity matrix:
       S1     S2     S3
S1:  1.000  0.898  0.727
S2:  0.898  1.000  0.722
S3:  0.727  0.722  1.000
```

## Go API

```go
import "github.com/rcarmo/gte-go/gte"

model, _ := gte.Load("gte-small.gtemodel")       // standard load
model, _ := gte.LoadMmap("gte-small.gtemodel")    // mmap: faster startup, less heap
defer model.Close()

emb, _ := model.Embed("Hello world")              // []float32, L2-normalized
batch, _ := model.EmbedBatch([]string{"hi", "there"})
sim, _ := gte.CosineSimilarity(batch[0], batch[1])
```

## Build Modes

| Mode | Command | Dependencies | Binary |
|---|---|---|---|
| **Pure Go + SIMD (default)** | `make` | None | Static, portable |
| OpenBLAS CGo | `CGO_ENABLED=1 make` | `gcc`, `libopenblas-dev` | Dynamic |

The default is `CGO_ENABLED=0`.  The OpenBLAS path calls `cblas_sgemm` via CGo for
AVX2 SIMD + zero allocations — yes, we cheated, but that's life.

### `Load()` vs `LoadMmap()`

| Build | `Load()` | `LoadMmap()` | Recommendation |
|---|---|---|---|
| Pure Go (amd64) | 10 ms, 0.25s startup | slower ⚠️ | **Use `Load()`** |
| Pure Go (arm64) | 20 ms, 0.25s startup | 20 ms, 0.01s startup | Either works |
| OpenBLAS CGo | 5.5 ms, 0.20s startup | 5.5 ms, 0.01s startup | **Use `LoadMmap()`** |

⚠️ `LoadMmap()` + gonum (amd64 NT path) triggers page faults from gonum's tiled read pattern.

## Testing

52 tests across all SIMD kernels, fast math, GEMM, and full embedding inference:

```bash
GTE_MODEL_PATH=gte-small.gtemodel go test ./...      # all tests
go test -bench=. -benchmem ./gte/simd/                # SIMD kernel benchmarks
make go-bench                                          # inference benchmark
```

## Optimization Report

![Optimization results](optimization-results.svg)

### Benchmark comparison — all configurations

#### amd64 — Intel i7-12700, 6 P-cores, 4.7GHz, Linux, Go 1.26.2

| Configuration | ms/embed | vs original | Allocs/op | B/op |
|---|---|---|---|---|
| Original (scalar loops, f64 math) | 67.4 | 1.0× | 170 | 15,598 |
| + float32 fast math | 63.4 | 1.06× | 170 | 15,598 |
| + gonum BLAS | 7.4 | 9.1× | 1,610 | 159,437 |
| + adaptive attention | 7.3 | 9.2× | 1,454 | 145,410 |
| + fused QKV + contiguous heads | 6.3 | 10.7× | 1,406 | 142,338 |
| **+ blocked FMA tile asm** | **10** | **6.7×** | **12** | **188** |
| + OpenBLAS CGo | **5.5** | **12.3×** | **12** | **186** |

#### arm64 — CIX P1 CD8160 (OrangePi 6 Plus), 12 cores, ~2GHz

| Configuration | ms/embed | vs gonum baseline | Allocs/op | B/op |
|---|---|---|---|---|
| gonum only (no SIMD) | 104 | 1.0× | 1,404 | 141,152 |
| **+ blocked NEON tile asm** | **20** | **5.2×** | **12** | **191** |

### Architecture-specific dispatch

The NT (NoTrans×Trans) matrix multiply — used by all `linear()` calls — is the
dominant bottleneck.  The optimal strategy differs by architecture:

| Path | amd64 | arm64 | Why |
|---|---|---|---|
| NT, small m (≤32) | gonum | GEBP + NEON asm | gonum has x86 asm `DotUnitary`; no NEON on arm64 |
| NT, large m (>32) | GEBP + AVX2 asm | GEBP + NEON asm | GEBP packing amortized over many tiles |
| NN (all sizes) | AVX2+FMA asm | NEON asm | Our tiled kernel with register-blocked C |
| Attention Q·K | AVX2 Sdot | NEON Sdot | SIMD dot product for score computation |

### What limits further improvement

**amd64**: gonum's `f32.DotUnitary` (used in NT for small m) does **not use FMA** —
it emits `VMULPS + VADDPS` (2 ops) instead of `VFMADD231PS` (1 op).  This accounts
for most of the 6.6ms→5.5ms gap to OpenBLAS.  Closing it requires writing the full
NT blocked loop in assembly (~300 lines).

**arm64**: The GEBP packing step transposes O(n×k) elements using scalar Go.
NEON-accelerated packing would save ~15ms.  The CIX P1 is also a low-power SoC
(~2GHz vs 4.7GHz on the i7).

### SIMD assembly files

| File | Architecture | Kernels |
|---|---|---|
| `gte/simd/simd_amd64.s` | x86-64 AVX2+FMA | `Sdot`, `Saxpy` |
| `gte/simd/simd_arm64.s` | ARM64 NEON | `Sdot`, `Saxpy` |
| `gte/simd/sgemm_amd64.s` | x86-64 AVX2+FMA | `SgemmNT`, `SgemmNN` |
| `gte/simd/sgemm_arm64.s` | ARM64 NEON | `SgemmNT`², `SgemmNN` |
| `gte/simd/gebp_amd64.s` | x86-64 AVX2+FMA | 6×16 GEBP micro-kernel |
| `gte/simd/gebp_arm64.s` | ARM64 NEON | 4×16 GEBP micro-kernel |
| `gte/simd/simd_other.go` | Scalar fallback | All kernels |

² Direct `SgemmNT` on arm64 has a known bug in the j-loop; not used in inference
(dispatch routes through GEBP).

### Other optimized files

- `gte/fastmath.go` — float32 tanh/exp/invSqrt (Padé/Newton, no float64)
- `gte/sgemm.go` — dispatch: OpenBLAS → SIMD asm → gonum fallback
- `gte/mmap.go` — `LoadMmap()` for memory-mapped model loading
- `gte/openblas_cgo.go` — direct CGo `cblas_sgemm` wrapper

## Model Format

`.gtemodel` is identical to the original C project: binary header, vocabulary,
and contiguous float32 weights.  Use `convert_model.py` to export from Hugging Face.

## License

MIT
