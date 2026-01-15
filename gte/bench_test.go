package gte

import (
	"os"
	"testing"
	"time"
)

// BenchmarkEmbed measures single-text embedding latency.
// Set GTE_MODEL_PATH to point at a .gtemodel file; otherwise defaults to gte-small.gtemodel.
func BenchmarkEmbed(b *testing.B) {
	path := os.Getenv("GTE_MODEL_PATH")
	if path == "" {
		path = "gte-small.gtemodel"
	}
	model, err := Load(path)
	if err != nil {
		b.Skipf("model load failed: %v", err)
	}
	defer model.Close()

	buf := make([]float32, model.Dim())
	text := "The stock market crashed"

	b.ReportAllocs()
	b.ResetTimer()
	start := time.Now()
	for i := 0; i < b.N; i++ {
		if err := model.EmbedTo(text, buf); err != nil {
			b.Fatalf("embed failed: %v", err)
		}
	}
	elapsed := time.Since(start)
	avg := float64(elapsed) / float64(b.N) / float64(time.Millisecond)
	b.ReportMetric(avg, "ms/op_avg")
}
