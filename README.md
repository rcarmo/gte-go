# GTE-Small in Go

A pure Go implementation of the [GTE-small](https://huggingface.co/thenlper/gte-small) text embedding model. Produces 384-dimensional, L2-normalized embeddings suitable for similarity search and clustering, directly ported from [@antirez's C implementation](https://github.com/antirez/gte-pure-C).

Performance is _not_ comparable to the C version, with embeddings generated ~3x slower due to Go's lack of low-level optimizations.

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

## Testing

```bash
GTE_MODEL_PATH=gte-small.gtemodel go test ./...
```

`gte/gte_test.go` embeds three reference sentences and checks cosine similarities within a small tolerance.

## License

MIT
