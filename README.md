# GTE-Small in Go

A pure Go implementation of the [GTE-small](https://huggingface.co/thenlper/gte-small) text embedding model. Produces 384-dimensional, L2-normalized embeddings suitable for similarity search and clustering, directly ported from [@antirez's C implementation](https://github.com/antirez/gte-pure-C).

Performance was originally ~3x slower than the C version due to Go's lack of low-level optimizations, but with BLAS-accelerated matrix operations and float32 fast-math approximations the gap has closed significantly (see [Optimization Log](#optimization-log) below).

## Quick Start

```bash
# Install deps for conversion
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

# Run the demo
go run ./cmd/gte --model-path gte-small.gtemodel "I love cats" "I love dogs" "The stock market crashed"

# Or via make
make                # builds Go binaries and runs tests
make run-go         # runs the demo with sample sentences
make run-bench      # runs the single-model benchmark (reports ms/op and throughput)
make go-bench       # go test benchmark (ms/op_avg via go test)
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

model, _ := gte.Load("gte-small.gtemodel")
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

## Optimization Log

All benchmarks on `12th Gen Intel Core i7-12700`, 6 P-cores, Linux/amd64, Go 1.26.2.
Text: `"The stock market crashed"` (5 tokens), `go test -bench=BenchmarkEmbed -benchtime=10s`.

| Stage | ms/op | vs baseline | Change |
|---|---|---|---|
| **Baseline** (main) | 67.4 | — | Manual loop unrolling, `math.Tanh/Exp/Sqrt` via float64 |
| **Phase 1a** — fast float32 math | 63.4 | 1.06× | Replace `math.Tanh/Exp/Sqrt` with float32 approximations |
| **Phase 1b** — gonum BLAS `linear()` | 7.4 | 9.1× | Replace hand-rolled dot products with `blas32.Sgemm` |
| **Phase 1c** — BLAS attention + scalar short-seq | 7.3 | 9.2× | BLAS Q·K^T and attn·V for long seqs; no goroutines for short seqs |
| **Phase 2a** — fused QKV projection | 6.6 | 10.2× | 3 Sgemm calls → 1 with concatenated Q/K/V weights |
| **Phase 2b** — contiguous head layout | 6.3 | 10.7× | QKV split writes directly to `[heads, seqLen, headDim]` layout |
| **Phase 2c** — fast GELU approximation | — | — | *skipped: sigmoid GELU too inaccurate for this model* |
| **Phase 2d** — zero-alloc serial sgemm | 6.5 | 10.4× | Custom serial matmul for small problems; gonum parallel for large |
| **Phase 3a** — OpenBLAS via CGo | 5.5 | 12.3× | Direct `cblas_sgemm` call, zero Go allocs, SIMD-accelerated |
| **Phase 3b** — zero-alloc tokenizer | 5.6 | 12.0× | Reuse tokenizer buffers, eliminate per-call slice allocations |

### Notes

- **gonum BLAS** (pure Go) accounts for ~85% of remaining allocations via internal goroutine spawning
- **OpenBLAS via CGo** (Phase 3) would eliminate those allocations and add SIMD, but `gonum.org/v1/netlib` has linking issues with Debian's OpenBLAS package
- **PGO** tested but within noise at current sizes
- Benchmark text: `"The stock market crashed"` (5 words → 7 tokens after CLS/SEP)

## License

MIT
