package main

import (
	"bufio"
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
	"time"

	"github.com/rcarmo/gte-go/gte"
)

func main() {
	modelPath := flag.String("model", "gte-small.gtemodel", "model path")
	textsPath := flag.String("texts", "", "file with one text per line")
	n := flag.Int("n", 5000, "number of embeddings")
	warmup := flag.Int("warmup", 50, "warmup embeddings")
	outPath := flag.String("out", "jitter.csv", "output CSV path")
	useMmap := flag.Bool("mmap", false, "use LoadMmap")
	flag.Parse()

	var model *gte.Model
	var err error
	if *useMmap {
		model, err = gte.LoadMmap(*modelPath)
	} else {
		model, err = gte.Load(*modelPath)
	}
	if err != nil {
		log.Fatalf("load: %v", err)
	}
	defer model.Close()

	// Load texts
	var texts []string
	if *textsPath != "" {
		f, err := os.Open(*textsPath)
		if err != nil {
			log.Fatalf("open texts: %v", err)
		}
		s := bufio.NewScanner(f)
		for s.Scan() {
			if t := s.Text(); t != "" {
				texts = append(texts, t)
			}
		}
		f.Close()
	}
	if len(texts) == 0 {
		texts = []string{
			"The quick brown fox jumps over the lazy dog",
			"Machine learning is transforming how we build software",
			"Soft drinks, coffees, teas, beers, and ales",
		}
	}

	buf := make([]float32, model.Dim())

	// Warmup
	for i := 0; i < *warmup; i++ {
		_ = model.EmbedTo(texts[i%len(texts)], buf)
	}

	// Measure
	latencies := make([]float64, *n)
	for i := 0; i < *n; i++ {
		text := texts[i%len(texts)]
		start := time.Now()
		if err := model.EmbedTo(text, buf); err != nil {
			log.Fatalf("embed %d: %v", i, err)
		}
		latencies[i] = float64(time.Since(start).Microseconds()) / 1000.0 // ms
	}

	// Write CSV
	out, err := os.Create(*outPath)
	if err != nil {
		log.Fatalf("create output: %v", err)
	}
	w := csv.NewWriter(out)
	w.Write([]string{"index", "ms"})
	for i, lat := range latencies {
		w.Write([]string{strconv.Itoa(i), fmt.Sprintf("%.3f", lat)})
	}
	w.Flush()
	out.Close()

	// Stats
	sum := 0.0
	min, max := latencies[0], latencies[0]
	for _, v := range latencies {
		sum += v
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	mean := sum / float64(len(latencies))
	variance := 0.0
	for _, v := range latencies {
		d := v - mean
		variance += d * d
	}
	stddev := 0.0
	if len(latencies) > 1 {
		stddev = variance / float64(len(latencies)-1)
		if stddev > 0 {
			stddev = float64(time.Duration(stddev*1e6).Seconds()) * 1000
		}
	}

	fmt.Printf("n=%d mean=%.3fms min=%.3fms max=%.3fms stddev=%.3fms\n",
		*n, mean, min, max, stddev)
	fmt.Printf("wrote %s\n", *outPath)
}
