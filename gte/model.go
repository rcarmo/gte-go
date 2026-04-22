package gte

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
)

const (
	fileMagic    = "GTE1"
	layerNormEps = 1e-12
	tokenPAD     = 0
	tokenUNK     = 100
	tokenCLS     = 101
	tokenSEP     = 102
	tokenMASK    = 103
)

// Model holds the loaded weights and working buffers.
type Model struct {
	VocabSize    int
	HiddenSize   int
	NumLayers    int
	NumHeads     int
	Intermediate int
	MaxSeqLen    int
	HeadDim      int

	Vocab    []string
	vocabMap map[string]int

	TokenEmbeddings []float32
	PositionEmb     []float32
	TokenTypeEmb    []float32
	EmbedLnWeight   []float32
	EmbedLnBias     []float32

	Layers       []LayerWeights
	PoolerWeight []float32
	PoolerBias   []float32

	hiddenStates []float32
	attnScores   []float32
	qProj        []float32
	kProj        []float32
	vProj        []float32
	attnOutput   []float32
	ffnHidden    []float32
	tempHidden   []float32
	qHeadBuf  []float32
	kHeadBuf  []float32
	vHeadBuf  []float32
	cHeadBuf  []float32
	tokenBuf     []int
	attnMaskBuf  []bool
}

// LayerWeights mirrors a transformer block.
type LayerWeights struct {
	QueryWeight      []float32
	QueryBias        []float32
	KeyWeight        []float32
	KeyBias          []float32
	ValueWeight      []float32
	ValueBias        []float32
	AttnOutputWeight []float32
	AttnOutputBias   []float32
	AttnLnWeight     []float32
	AttnLnBias       []float32

	FfnInterWeight  []float32
	FfnInterBias    []float32
	FfnOutputWeight []float32
	FfnOutputBias   []float32
	FfnLnWeight     []float32
	FfnLnBias       []float32
}

// Load reads a .gtemodel file into memory.
func Load(modelPath string) (*Model, error) {
	f, err := os.Open(modelPath)
	if err != nil {
		return nil, fmt.Errorf("open model: %w", err)
	}
	defer f.Close()

	r := bufio.NewReader(f)

	magic := make([]byte, 4)
	if _, err := io.ReadFull(r, magic); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if string(magic) != fileMagic {
		return nil, errors.New("invalid model magic")
	}

	cfg := make([]int, 6)
	for i := 0; i < 6; i++ {
		v, err := readUint32(r)
		if err != nil {
			return nil, fmt.Errorf("read config %d: %w", i, err)
		}
		cfg[i] = v
	}

	m := &Model{
		VocabSize:    cfg[0],
		HiddenSize:   cfg[1],
		NumLayers:    cfg[2],
		NumHeads:     cfg[3],
		Intermediate: cfg[4],
		MaxSeqLen:    cfg[5],
	}
	if m.HiddenSize%m.NumHeads != 0 {
		return nil, fmt.Errorf("hidden_size %% num_heads != 0: %d %% %d", m.HiddenSize, m.NumHeads)
	}
	m.HeadDim = m.HiddenSize / m.NumHeads

	if err := m.readVocab(r); err != nil {
		return nil, err
	}
	if err := m.readEmbeddings(r); err != nil {
		return nil, err
	}
	if err := m.readLayers(r); err != nil {
		return nil, err
	}
	if err := m.readPooler(r); err != nil {
		return nil, err
	}
	if err := m.initBuffers(); err != nil {
		return nil, err
	}

	return m, nil
}

func (m *Model) readVocab(r *bufio.Reader) error {
	m.Vocab = make([]string, m.VocabSize)
	m.vocabMap = make(map[string]int, m.VocabSize)
	for i := 0; i < m.VocabSize; i++ {
		length, err := readUint16(r)
		if err != nil {
			return fmt.Errorf("read vocab len %d: %w", i, err)
		}
		buf := make([]byte, length)
		if _, err := io.ReadFull(r, buf); err != nil {
			return fmt.Errorf("read vocab %d: %w", i, err)
		}
		m.Vocab[i] = string(buf)
		m.vocabMap[m.Vocab[i]] = i
	}
	return nil
}

func (m *Model) readEmbeddings(r *bufio.Reader) error {
	var err error
	if m.TokenEmbeddings, err = readFloat32Slice(r, m.VocabSize*m.HiddenSize); err != nil {
		return fmt.Errorf("token embeddings: %w", err)
	}
	if m.PositionEmb, err = readFloat32Slice(r, m.MaxSeqLen*m.HiddenSize); err != nil {
		return fmt.Errorf("position embeddings: %w", err)
	}
	if m.TokenTypeEmb, err = readFloat32Slice(r, 2*m.HiddenSize); err != nil {
		return fmt.Errorf("token type embeddings: %w", err)
	}
	if m.EmbedLnWeight, err = readFloat32Slice(r, m.HiddenSize); err != nil {
		return fmt.Errorf("embed ln weight: %w", err)
	}
	if m.EmbedLnBias, err = readFloat32Slice(r, m.HiddenSize); err != nil {
		return fmt.Errorf("embed ln bias: %w", err)
	}
	return nil
}

func (m *Model) readLayers(r *bufio.Reader) error {
	m.Layers = make([]LayerWeights, m.NumLayers)
	for l := 0; l < m.NumLayers; l++ {
		lw := &m.Layers[l]
		var err error
		if lw.QueryWeight, err = readFloat32Slice(r, m.HiddenSize*m.HiddenSize); err != nil {
			return fmt.Errorf("layer %d query weight: %w", l, err)
		}
		if lw.QueryBias, err = readFloat32Slice(r, m.HiddenSize); err != nil {
			return fmt.Errorf("layer %d query bias: %w", l, err)
		}
		if lw.KeyWeight, err = readFloat32Slice(r, m.HiddenSize*m.HiddenSize); err != nil {
			return fmt.Errorf("layer %d key weight: %w", l, err)
		}
		if lw.KeyBias, err = readFloat32Slice(r, m.HiddenSize); err != nil {
			return fmt.Errorf("layer %d key bias: %w", l, err)
		}
		if lw.ValueWeight, err = readFloat32Slice(r, m.HiddenSize*m.HiddenSize); err != nil {
			return fmt.Errorf("layer %d value weight: %w", l, err)
		}
		if lw.ValueBias, err = readFloat32Slice(r, m.HiddenSize); err != nil {
			return fmt.Errorf("layer %d value bias: %w", l, err)
		}
		if lw.AttnOutputWeight, err = readFloat32Slice(r, m.HiddenSize*m.HiddenSize); err != nil {
			return fmt.Errorf("layer %d attn output weight: %w", l, err)
		}
		if lw.AttnOutputBias, err = readFloat32Slice(r, m.HiddenSize); err != nil {
			return fmt.Errorf("layer %d attn output bias: %w", l, err)
		}
		if lw.AttnLnWeight, err = readFloat32Slice(r, m.HiddenSize); err != nil {
			return fmt.Errorf("layer %d attn ln weight: %w", l, err)
		}
		if lw.AttnLnBias, err = readFloat32Slice(r, m.HiddenSize); err != nil {
			return fmt.Errorf("layer %d attn ln bias: %w", l, err)
		}

		if lw.FfnInterWeight, err = readFloat32Slice(r, m.Intermediate*m.HiddenSize); err != nil {
			return fmt.Errorf("layer %d ffn inter weight: %w", l, err)
		}
		if lw.FfnInterBias, err = readFloat32Slice(r, m.Intermediate); err != nil {
			return fmt.Errorf("layer %d ffn inter bias: %w", l, err)
		}
		if lw.FfnOutputWeight, err = readFloat32Slice(r, m.HiddenSize*m.Intermediate); err != nil {
			return fmt.Errorf("layer %d ffn output weight: %w", l, err)
		}
		if lw.FfnOutputBias, err = readFloat32Slice(r, m.HiddenSize); err != nil {
			return fmt.Errorf("layer %d ffn output bias: %w", l, err)
		}
		if lw.FfnLnWeight, err = readFloat32Slice(r, m.HiddenSize); err != nil {
			return fmt.Errorf("layer %d ffn ln weight: %w", l, err)
		}
		if lw.FfnLnBias, err = readFloat32Slice(r, m.HiddenSize); err != nil {
			return fmt.Errorf("layer %d ffn ln bias: %w", l, err)
		}
	}
	return nil
}

func (m *Model) readPooler(r *bufio.Reader) error {
	var err error
	if m.PoolerWeight, err = readFloat32Slice(r, m.HiddenSize*m.HiddenSize); err != nil {
		return fmt.Errorf("pooler weight: %w", err)
	}
	if m.PoolerBias, err = readFloat32Slice(r, m.HiddenSize); err != nil {
		return fmt.Errorf("pooler bias: %w", err)
	}
	return nil
}

func (m *Model) initBuffers() error {
	maxSeq := m.MaxSeqLen
	hidden := m.HiddenSize
	inter := m.Intermediate
	heads := m.NumHeads

	m.hiddenStates = make([]float32, maxSeq*hidden)
	m.attnScores = make([]float32, heads*maxSeq*maxSeq)
	m.qProj = make([]float32, maxSeq*hidden)
	m.kProj = make([]float32, maxSeq*hidden)
	m.vProj = make([]float32, maxSeq*hidden)
	m.attnOutput = make([]float32, maxSeq*hidden)
	m.ffnHidden = make([]float32, maxSeq*inter)
	m.tempHidden = make([]float32, maxSeq*hidden)
	m.qHeadBuf = make([]float32, heads*maxSeq*m.HeadDim)
	m.kHeadBuf = make([]float32, heads*maxSeq*m.HeadDim)
	m.vHeadBuf = make([]float32, heads*maxSeq*m.HeadDim)
	m.cHeadBuf = make([]float32, heads*maxSeq*m.HeadDim)

	return nil
}

// Dim returns the embedding dimension.
func (m *Model) Dim() int { return m.HiddenSize }

// MaxLen returns the maximum sequence length.
func (m *Model) MaxLen() int { return m.MaxSeqLen }

// Close clears references for GC.
func (m *Model) Close() {
	*m = Model{}
}

func readUint32(r io.Reader) (int, error) {
	var v uint32
	if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
		return 0, err
	}
	return int(v), nil
}

func readUint16(r io.Reader) (int, error) {
	var v uint16
	if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
		return 0, err
	}
	return int(v), nil
}

func readFloat32Slice(r io.Reader, count int) ([]float32, error) {
	buf := make([]float32, count)
	if err := binary.Read(r, binary.LittleEndian, buf); err != nil {
		return nil, err
	}
	return buf, nil
}
