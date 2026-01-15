GO ?= go
GOFLAGS ?= -trimpath
PGO_PROFILE ?= cpu.pprof

ifeq (,$(wildcard $(PGO_PROFILE)))
PGO_FLAG :=
else
PGO_FLAG := -pgo=$(PGO_PROFILE)
endif

all: go-build go-test

go-build:
	$(GO) build $(GOFLAGS) $(PGO_FLAG) ./cmd/gte ./cmd/test_gte ./cmd/bench

go-test: gte-small.gtemodel
	GTE_MODEL_PATH=gte-small.gtemodel $(GO) test $(GOFLAGS) $(PGO_FLAG) ./...

run-go: gte-small.gtemodel
	$(GO) run $(GOFLAGS) $(PGO_FLAG) ./cmd/gte --model-path gte-small.gtemodel "I love cats" "I love dogs" "The stock market crashed"

run-bench: gte-small.gtemodel
	$(GO) run $(GOFLAGS) $(PGO_FLAG) ./cmd/bench --model-path gte-small.gtemodel --iterations 100 --warmup 5

bench-profile: gte-small.gtemodel
	$(GO) run $(GOFLAGS) ./cmd/bench --model-path gte-small.gtemodel --iterations 200 --warmup 10 --cpuprofile $(PGO_PROFILE)

go-build-pgo: $(PGO_PROFILE)
	$(GO) build $(GOFLAGS) -pgo=$(PGO_PROFILE) ./cmd/gte ./cmd/test_gte ./cmd/bench

gte-small.gtemodel: models/gte-small/model.safetensors
	python3 convert_model.py models/gte-small $@

clean:
	rm -f gte-small.gtemodel $(PGO_PROFILE)

.PHONY: all clean go-build go-test run-go run-bench bench-profile go-build-pgo

go-bench: gte-small.gtemodel
	GTE_MODEL_PATH=gte-small.gtemodel $(GO) test $(GOFLAGS) $(PGO_FLAG) -bench=BenchmarkEmbed -benchmem ./gte
