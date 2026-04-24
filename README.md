# GTE-Small in Go

A pure Go implementation of the [GTE-small](https://huggingface.co/thenlper/gte-small) text embedding model. Produces 384-dimensional, L2-normalized embeddings suitable for similarity search and clustering, ported from [@antirez's C implementation](https://github.com/antirez/gte-pure-C).

**Single static binary. Zero allocations in the hot path. Predictable latency.**

| Platform | ms/embed | Allocs | GC pressure |
|---|---|---|---|
| **amd64** (i7-12700) | **11 ms** | **1** | **~700 B/s @ 100 qps** |
| **arm64** (CIX P1 CD8160) | **20 ms** | **1** | **~700 B/s @ 50 qps** |
| amd64 OpenBLAS CGo (opt-in) | 5.5 ms | 12 | ~700 B/s @ 180 qps |

The default build produces a **fully self-contained static binary** with no C dependencies, no gonum in the hot path, and no goroutine churn. All matrix operations use hand-written SIMD assembly (AVX2+FMA on amd64, NEON on arm64). This makes latency flat and predictable — no GC pauses, no goroutine scheduling jitter.

![Optimization results](optimization-results.svg)

## Why low-latency matters

Embedding models are often called inline during search, retrieval, or classification. Goroutine-heavy BLAS implementations create **13 MB/s of garbage at 100 queries/sec** (1,404 allocs × 141 KB per embedding), causing GC pauses that spike p99 latency. Our SIMD assembly path generates **10,000× less garbage** (1 alloc × 187 bytes), keeping GC pressure under 20 KB/s even at high throughput.

| | gonum BLAS | This project |
|---|---|---|
| Allocs/embed | 1,404 | **1** |
| Bytes/embed | 141 KB | **7 B** |
| Goroutine churn | 140K/s @ 100 qps | **0** |
| GC pressure | 13.4 MB/s | **~700 B/s** |
| Latency jitter | GC pauses | **None** |

## Quick Start

```bash
pip install safetensors requests numpy
python convert_model.py models/gte-small gte-small.gtemodel
make run-go                          # pure Go (default)
CGO_ENABLED=1 make run-go           # with OpenBLAS (max throughput)
```

## API

```go
import "github.com/rcarmo/gte-go/gte"

model, _ := gte.Load("gte-small.gtemodel")       // or LoadMmap() for fast startup
defer model.Close()

emb, _ := model.Embed("Hello world")              // []float32, L2-normalized
batch, _ := model.EmbedBatch([]string{"hi", "there"})
batch, _ = model.EmbedBatchParallel(texts, 0)     // concurrent with N workers
sim, _ := gte.CosineSimilarity(batch[0], batch[1])
```

## Build

| Mode | Command | Latency | Binary |
|---|---|---|---|
| **Pure Go + SIMD (default)** | `make` | Flat, predictable | Static, portable |
| OpenBLAS CGo | `CGO_ENABLED=1 make` | Lower avg, higher p99 | Dynamic |

Default is `CGO_ENABLED=0`. Use `Load()` for pure Go builds, `LoadMmap()` for OpenBLAS.

## Testing

```bash
GTE_MODEL_PATH=gte-small.gtemodel go test ./...   # 52+ tests
make go-bench                                       # inference benchmark
```

## Documentation

- **[docs/optimization-report.md](docs/optimization-report.md)** — 4-phase optimization from 67ms to 5.5ms
- **[docs/simd-assembly.md](docs/simd-assembly.md)** — SIMD kernel reference and register conventions
- **[docs/benchmarks.md](docs/benchmarks.md)** — cross-platform benchmarks and profile breakdown

## Architecture

All matrix operations in the inference hot path use hand-written assembly:

| Kernel | amd64 | arm64 | Technique |
|---|---|---|---|
| NT matmul | VGATHERDPS 6×8 | GEBP NEON 4×16 | No packing, no horizontal reductions |
| NN matmul | AVX2+FMA 32-wide | NEON 16-wide | Register-tiled C accumulation |
| Dot product | AVX2 FMA | NEON VFMLA | Attention score computation |
| B-panel pack | — | NEON 4×4 transpose | GEBP column-panel format |

Zero gonum dependency in the hot path. Zero goroutine spawning. 12 allocations per embedding (tokenizer string ops only).

## Model Format

`.gtemodel` — binary header + vocabulary + contiguous float32 weights.
Use `convert_model.py` to export from Hugging Face.

## License

MIT
