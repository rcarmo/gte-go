# GTE-Small in Go

A pure Go implementation of the [GTE-small](https://huggingface.co/thenlper/gte-small) text embedding model. Produces 384-dimensional, L2-normalized embeddings suitable for similarity search and clustering, directly ported from [@antirez's C implementation](https://github.com/antirez/gte-pure-C).

The default build is **pure Go** — a single static binary with no C dependencies — and runs at **~14ms per embedding** (5× faster than the original). For maximum throughput, an optional OpenBLAS CGo backend pushes that to **~7ms** (see [Optimization Report](#optimization-report) below).

## Quick Start

```bash
# Install deps for model conversion
pip install safetensors requests

# Download Hugging Face weights and convert to .gtemodel
python - <<'PY'
import os, requests, pathlib
base = "https://huggingface.co/thenlper/gte-small/resolve/main"
files = ["config.json", "vocab.txt", "tokenizer_config.json", "special_tokens_map.json", "model.safetensors"]
out = pathlib.Path("models/gte-small")
out.mkdir(parents=True, exist_ok=True)
for name in files:
    url = f"{base}/{name}"
    path = out / name
    if path.exists():
        continue
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
PY
python convert_model.py models/gte-small gte-small.gtemodel

# Build and run (pure Go, no C dependencies)
make run-go

# Or with OpenBLAS for ~2× faster inference (requires gcc + libopenblas-dev)
CGO_ENABLED=1 make run-go
```

Sample output:

```bash
Model loaded in 0.11 s
Embedding dimension: 384
Max sequence length: 512

Cosine similarity matrix:
       S1     S2     S3
S1:  1.000  0.898  0.727
S2:  0.898  1.000  0.722
S3:  0.727  0.722  1.000
```

## Go API

```go
import "github.com/rcarmo/gte-go/gte"

// Standard loading (copies weights into Go heap)
model, _ := gte.Load("gte-small.gtemodel")
defer model.Close()

// Memory-mapped loading (6× faster startup, ~127MB less heap)
model, _ := gte.LoadMmap("gte-small.gtemodel")
defer model.Close()

emb, _ := model.Embed("Hello world")          // []float32 length 384, L2-normalized
embBatch, _ := model.EmbedBatch([]string{"hi", "there"})
sim, _ := gte.CosineSimilarity(embBatch[0], embBatch[1])
```

## Model Format

`.gtemodel` is identical to the original C project: a binary header, vocabulary, and contiguous float32 weights. Use `convert_model.py` to export from Hugging Face weights.

## Testing & Benchmarks

```bash
GTE_MODEL_PATH=gte-small.gtemodel go test ./...
GTE_MODEL_PATH=gte-small.gtemodel go test -bench=BenchmarkEmbed -benchmem ./gte
make run-bench   # convenient single-model benchmark with human-readable output
```

- `gte/gte_test.go` embeds three reference sentences and checks cosine similarities within a small tolerance.
- `gte/bench_test.go` reports per-embedding latency (ms/op_avg) via `go test`.
- `cmd/bench` prints total calls, average ms per embedding (derived from total_time/total_calls), and throughput.

## Optimization Report

The original implementation used hand-unrolled scalar loops for all matrix operations
and Go's `math.Tanh/Exp/Sqrt` functions (which round-trip through float64). A
systematic 3-phase optimization brought inference latency from **67.4ms to 5.5ms**
— a **12.3× speedup** — while reducing per-embedding allocations from 170/15.6KB
to 12/186B.

![Optimization results](optimization-results.svg)

### Phase 1 — BLAS and fast math (67.4 → 7.3ms, 9.2×)

- **1a – float32 fast math:** Replaced `math.Tanh`, `math.Exp`, and `math.Sqrt`
  (all float64) with float32 polynomial/Padé approximations (`fastTanh`,
  `fastExp`, `fastInvSqrt`). Modest 6% gain because `linear()` dominates.
- **1b – gonum BLAS:** Replaced the hand-rolled `linear()` dot-product loops
  with `blas32.Sgemm` from gonum. This was the single largest improvement:
  **67.4 → 7.4ms (9.1×)**. Even gonum's pure-Go BLAS with cache blocking
  massively outperforms scalar code.
- **1c – adaptive attention:** Added a seqLen-based threshold: short sequences
  (< 32 tokens) use a scalar attention path without goroutine overhead;
  longer sequences use BLAS for Q·K^T and attn·V.

### Phase 2 — structural optimizations (7.3 → 6.3ms, 10.7×)

- **2a – fused QKV projection:** Concatenated Q/K/V weight matrices into a
  single `[3*hidden, hidden]` array at load time. One Sgemm call replaces three,
  reducing call overhead and improving cache utilization.
- **2b – contiguous head layout:** Changed the QKV split to write directly into
  `[numHeads, seqLen, headDim]` buffers instead of the interleaved
  `[seqLen, numHeads*headDim]` layout. Eliminates strided access in the
  attention inner loops and removes the de-interleave copy for the BLAS path.
- **2c – sigmoid GELU:** Tested but abandoned — the approximation error
  exceeded the model's cosine similarity tolerance by 0.006.

### Phase 3 — system-level optimizations (6.3 → 5.5ms, 12.3×)

- **3a – OpenBLAS via CGo:** Wrote a direct CGo wrapper for `cblas_sgemm`,
  bypassing gonum entirely. This eliminated **99% of per-embedding allocations**
  (1406 → 14) because gonum's pure-Go `sgemmParallel` spawns goroutines
  internally. OpenBLAS also provides AVX2 SIMD acceleration.
- **3b – zero-alloc tokenizer:** Refactored `basicTokenize` and
  `wordpieceTokenize` to reuse pre-allocated buffers instead of allocating
  slices per call. Reduced remaining allocs from 14 → 12, 1.4KB → 186B.
- **3c – mmap model loading:** Added `LoadMmap()` which memory-maps the
  `.gtemodel` file. Weight slices point directly into the mapped region —
  no copy into Go heap. Result: **6× faster startup** (0.03s vs 0.18s)
  and ~127MB less resident memory. Inference latency is unchanged.

### Results summary

#### Final builds

| Build | Load method | Median ms/embed | vs baseline | Allocs/op | B/op |
|---|---|---|---|---|---|
| Baseline (before optimization) | `Load()` | 67.4 | — | 170 | 15,598 |
| **Pure Go (default)** | **`Load()`** | **13.9** | **4.9×** | **12** | **186** |
| OpenBLAS CGo (opt-in) | `Load()` | 7.4 | 9.1× | 12 | 186 |
| OpenBLAS CGo (opt-in) | `LoadMmap()` | 7.2 | 9.4× | 12 | 186 |
| Pure Go | `LoadMmap()` | 34.4 | 2.0× ⚠️ | 12 | 186 |

⚠️ `LoadMmap()` + pure Go is slow because gonum's tiled BLAS triggers page faults on mmap'd memory.

#### Optimization stages (single-text benchmark)

| Stage | ms/op | vs baseline | Change |
|---|---|---|---|
| **Baseline** | 67.4 | — | Manual loop unrolling, `math.Tanh/Exp/Sqrt` via float64 |
| Phase 1a – fast math | 63.4 | 1.06× | float32 tanh/exp/invSqrt approximations |
| Phase 1b – gonum BLAS | 7.4 | 9.1× | `blas32.Sgemm` replaces hand-rolled dot products |
| Phase 1c – adaptive attn | 7.3 | 9.2× | No goroutines for short sequences |
| Phase 2a – fused QKV | 6.6 | 10.2× | 3 Sgemm → 1 with concatenated weights |
| Phase 2b – head layout | 6.3 | 10.7× | Contiguous `[heads, seqLen, headDim]` buffers |
| Phase 3a – OpenBLAS CGo | 5.5 | 12.3× | Direct `cblas_sgemm`, AVX2 SIMD, zero Go allocs |
| Phase 3b – zero-alloc tok | 5.6 | 12.0× | Reuse tokenizer buffers |
| Phase 3c – mmap loading | 5.6 | 12.0× | 6× faster startup, 127MB less heap |

All benchmarks on 12th Gen Intel Core i7-12700, 6 P-cores, Linux/amd64, Go 1.26.2.
Multi-text benchmarks: 15 diverse sentences (1–44 words, inc. unicode), 20 samples, median.

### Files added

- `gte/fastmath.go` — float32 approximations for tanh, exp, inverse sqrt
- `gte/sgemm.go` — matmul dispatch: OpenBLAS (CGo) or gonum (pure Go)
- `gte/openblas_cgo.go` — direct CGo wrapper for `cblas_sgemm` (built only with `CGO_ENABLED=1`)
- `gte/openblas_nocgo.go` — stub for pure-Go builds (`CGO_ENABLED=0`)
- `gte/mmap.go` — memory-mapped model loading via `LoadMmap()`
- `cmd/bench_compare/` — multi-text benchmark comparing Load vs LoadMmap
- `optimization-results.svg` — benchmark results chart

### Build modes

| Mode | Command | ms/embed | Dependencies | Binary |
|---|---|---|---|---|
| **Pure Go (default)** | `make` | ~14 | None | Static, portable |
| OpenBLAS CGo | `CGO_ENABLED=1 make` or `make go-build-cgo` | ~7 | `gcc`, `libopenblas-dev` | Dynamic, links libopenblas |

The default is **`CGO_ENABLED=0`** (set in the Makefile).
This produces a single static binary that runs anywhere with no system
library dependencies.  The pure-Go path uses gonum's cache-blocked BLAS
implementation, which is already 5× faster than the original baseline.

For maximum throughput, opt in to OpenBLAS:

```bash
sudo apt install gcc libopenblas-dev   # Debian/Ubuntu
CGO_ENABLED=1 make                     # or: make go-build-cgo
```

The OpenBLAS path calls `cblas_sgemm` directly via CGo — yes, we cheated,
but that's life.  It adds AVX2 SIMD and eliminates the goroutine allocations
that gonum's parallel BLAS introduces (~1400 allocs/embed → 12).

### `Load()` vs `LoadMmap()` guidance

| Build | `Load()` | `LoadMmap()` | Recommendation |
|---|---|---|---|
| **Pure Go** | 13.9 ms, 0.24s startup | 34.4 ms, 0.01s startup | **Use `Load()`** |
| **OpenBLAS CGo** | 7.4 ms, 0.20s startup | 7.2 ms, 0.01s startup | **Use `LoadMmap()`** |

`LoadMmap()` maps the model file directly — 12× faster startup and ~127MB
less heap.  With OpenBLAS this is free performance.  With pure Go, gonum's
cache-blocked BLAS reads weight data in a scattered tiled pattern that
triggers constant page faults on mmap'd memory, making it 2.5× slower
than `Load()`.  Use `Load()` for pure-Go builds.

## License

MIT
