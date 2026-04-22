package gte

import (
	"errors"
	"fmt"
	"sync"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
	"gonum.org/v1/gonum/blas/gonum"
)

func init() {
	blas32.Use(gonum.Implementation{})
}

// linear computes Y = X·W^T + bias using BLAS GEMM.
// W is stored as [outDim, inDim] (row-major), so W^T gives us X·W^T = [seqLen, outDim].
func linear(y, x, w, b []float32, seqLen, inDim, outDim int) {
	// Y[seqLen, outDim] = X[seqLen, inDim] · W^T[inDim, outDim]
	// GEMM: C = alpha*A*B + beta*C
	// With B = W (row-major [outDim, inDim]), transposed => [inDim, outDim]
	blas32.Implementation().Sgemm(
		blas.NoTrans, blas.Trans,
		seqLen, outDim, inDim,
		1.0,
		x, inDim,  // A = X [seqLen x inDim]
		w, inDim,  // B = W [outDim x inDim], transposed
		0.0,
		y, outDim, // C = Y [seqLen x outDim]
	)

	// Add bias
	if b != nil {
		for s := 0; s < seqLen; s++ {
			yRow := y[s*outDim:][:outDim]
			for o := 0; o < outDim; o++ {
				yRow[o] += b[o]
			}
		}
	}
}

func layerNorm(out, x, gamma, beta []float32, seqLen, hidden int) {
	for s := 0; s < seqLen; s++ {
		rowStart := s * hidden
		mean := float32(0)
		for i := 0; i < hidden; i++ {
			mean += x[rowStart+i]
		}
		mean /= float32(hidden)

		variance := float32(0)
		for i := 0; i < hidden; i++ {
			diff := x[rowStart+i] - mean
			variance += diff * diff
		}
		variance /= float32(hidden)
		stdInv := fastInvSqrt(variance + layerNormEps)

		for i := 0; i < hidden; i++ {
			out[rowStart+i] = gamma[i]*(x[rowStart+i]-mean)*stdInv + beta[i]
		}
	}
}

func gelu(x []float32) {
	const c = float32(0.7978845608) // sqrt(2/pi)
	for i := range x {
		v := x[i]
		x[i] = 0.5 * v * (1 + fastTanh(c*(v+0.044715*v*v*v)))
	}
}

func softmax(x []float32) {
	maxVal := x[0]
	for _, v := range x[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	sum := float32(0)
	for i, v := range x {
		x[i] = fastExp(v - maxVal)
		sum += x[i]
	}
	inv := 1 / sum
	for i := range x {
		x[i] *= inv
	}
}

func l2Normalize(x []float32) {
	norm := float32(0)
	for _, v := range x {
		norm += v * v
	}
	if norm == 0 {
		return
	}
	inv := fastInvSqrt(norm)
	for i := range x {
		x[i] *= inv
	}
}

func (m *Model) selfAttention(layer *LayerWeights, seqLen int, attnMask []bool) {
	hidden := m.HiddenSize
	heads := m.NumHeads
	headDim := m.HeadDim

	linear(m.qProj, m.hiddenStates, layer.QueryWeight, layer.QueryBias, seqLen, hidden, hidden)
	linear(m.kProj, m.hiddenStates, layer.KeyWeight, layer.KeyBias, seqLen, hidden, hidden)
	linear(m.vProj, m.hiddenStates, layer.ValueWeight, layer.ValueBias, seqLen, hidden, hidden)

	scale := fastInvSqrt(float32(headDim))

	// For short sequences, goroutine overhead exceeds parallelism benefit.
	if seqLen < 32 {
		for h := 0; h < heads; h++ {
			m.selfAttentionHeadScalar(h, seqLen, headDim, hidden, scale, attnMask)
		}
	} else {
		var wg sync.WaitGroup
		wg.Add(heads)
		for h := 0; h < heads; h++ {
			h := h
			go func() {
				m.selfAttentionHeadBLAS(h, seqLen, headDim, hidden, scale, attnMask)
				wg.Done()
			}()
		}
		wg.Wait()
	}

	linear(m.tempHidden, m.attnOutput, layer.AttnOutputWeight, layer.AttnOutputBias, seqLen, hidden, hidden)
	for i := 0; i < seqLen*hidden; i++ {
		m.tempHidden[i] += m.hiddenStates[i]
	}
	layerNorm(m.hiddenStates, m.tempHidden, layer.AttnLnWeight, layer.AttnLnBias, seqLen, hidden)
}

// selfAttentionHeadScalar is the original scalar path, fast for short sequences.
func (m *Model) selfAttentionHeadScalar(h, seqLen, headDim, hidden int, scale float32, attnMask []bool) {
	headOffset := h * seqLen * seqLen
	for i := 0; i < seqLen; i++ {
		rowOffset := headOffset + i*seqLen
		for j := 0; j < seqLen; j++ {
			score := float32(0)
			for d := 0; d < headDim; d++ {
				qIdx := i*hidden + h*headDim + d
				kIdx := j*hidden + h*headDim + d
				score += m.qProj[qIdx] * m.kProj[kIdx]
			}
			score *= scale
			if attnMask != nil && !attnMask[j] {
				score = -10000
			}
			m.attnScores[rowOffset+j] = score
		}
		softmax(m.attnScores[rowOffset : rowOffset+seqLen])
	}

	for i := 0; i < seqLen; i++ {
		for d := 0; d < headDim; d++ {
			sum := float32(0)
			for j := 0; j < seqLen; j++ {
				attn := m.attnScores[headOffset+i*seqLen+j]
				vIdx := j*hidden + h*headDim + d
				sum += attn * m.vProj[vIdx]
			}
			m.attnOutput[i*hidden+h*headDim+d] = sum
		}
	}
}

// selfAttentionHeadBLAS uses Sgemm for Q·K^T and attn·V, better for longer sequences.
func (m *Model) selfAttentionHeadBLAS(h, seqLen, headDim, hidden int, scale float32, attnMask []bool) {
	off := h * seqLen * headDim
	qBuf := m.qHeadBuf[off : off+seqLen*headDim]
	kBuf := m.kHeadBuf[off : off+seqLen*headDim]
	vBuf := m.vHeadBuf[off : off+seqLen*headDim]
	cBuf := m.cHeadBuf[off : off+seqLen*headDim]

	// De-interleave: [seqLen, hidden] -> per-head [seqLen, headDim]
	for s := 0; s < seqLen; s++ {
		srcOff := s*hidden + h*headDim
		dstOff := s * headDim
		copy(qBuf[dstOff:dstOff+headDim], m.qProj[srcOff:srcOff+headDim])
		copy(kBuf[dstOff:dstOff+headDim], m.kProj[srcOff:srcOff+headDim])
		copy(vBuf[dstOff:dstOff+headDim], m.vProj[srcOff:srcOff+headDim])
	}

	// scores[seqLen, seqLen] = Q[seqLen, headDim] · K^T[headDim, seqLen]
	scoreOff := h * seqLen * seqLen
	scores := m.attnScores[scoreOff : scoreOff+seqLen*seqLen]
	blas32.Implementation().Sgemm(
		blas.NoTrans, blas.Trans,
		seqLen, seqLen, headDim,
		scale,
		qBuf, headDim,
		kBuf, headDim,
		0.0,
		scores, seqLen,
	)

	// Apply mask + softmax per row
	for i := 0; i < seqLen; i++ {
		row := scores[i*seqLen : i*seqLen+seqLen]
		if attnMask != nil {
			for j := 0; j < seqLen; j++ {
				if !attnMask[j] {
					row[j] = -10000
				}
			}
		}
		softmax(row)
	}

	// context[seqLen, headDim] = scores[seqLen, seqLen] · V[seqLen, headDim]
	blas32.Implementation().Sgemm(
		blas.NoTrans, blas.NoTrans,
		seqLen, headDim, seqLen,
		1.0,
		scores, seqLen,
		vBuf, headDim,
		0.0,
		cBuf, headDim,
	)

	// Re-interleave: per-head [seqLen, headDim] -> [seqLen, hidden]
	for s := 0; s < seqLen; s++ {
		srcOff := s * headDim
		dstOff := s*hidden + h*headDim
		copy(m.attnOutput[dstOff:dstOff+headDim], cBuf[srcOff:srcOff+headDim])
	}
}

func (m *Model) feedForward(layer *LayerWeights, seqLen int) {
	hidden := m.HiddenSize
	inter := m.Intermediate

	linear(m.ffnHidden, m.hiddenStates, layer.FfnInterWeight, layer.FfnInterBias, seqLen, hidden, inter)
	gelu(m.ffnHidden[:seqLen*inter])

	linear(m.tempHidden, m.ffnHidden, layer.FfnOutputWeight, layer.FfnOutputBias, seqLen, inter, hidden)
	for i := 0; i < seqLen*hidden; i++ {
		m.tempHidden[i] += m.hiddenStates[i]
	}
	layerNorm(m.hiddenStates, m.tempHidden, layer.FfnLnWeight, layer.FfnLnBias, seqLen, hidden)
}

func (m *Model) transformerForward(tokenIDs []int, seqLen int, attnMask []bool) {
	hidden := m.HiddenSize
	for s := 0; s < seqLen; s++ {
		tokenID := tokenIDs[s]
		base := s * hidden
		embOffset := tokenID * hidden
		posOffset := s * hidden
		for d := 0; d < hidden; d++ {
			m.hiddenStates[base+d] = m.TokenEmbeddings[embOffset+d] + m.PositionEmb[posOffset+d] + m.TokenTypeEmb[d]
		}
	}
	layerNorm(m.hiddenStates, m.hiddenStates, m.EmbedLnWeight, m.EmbedLnBias, seqLen, hidden)

	for l := 0; l < m.NumLayers; l++ {
		m.selfAttention(&m.Layers[l], seqLen, attnMask)
		m.feedForward(&m.Layers[l], seqLen)
	}
}

func meanPooling(output, hiddenStates []float32, attnMask []bool, seqLen, hidden int) {
	for i := range output {
		output[i] = 0
	}
	count := 0
	for s := 0; s < seqLen; s++ {
		if attnMask[s] {
			base := s * hidden
			for d := 0; d < hidden; d++ {
				output[d] += hiddenStates[base+d]
			}
			count++
		}
	}
	if count == 0 {
		return
	}
	inv := float32(1.0 / float64(count))
	for d := 0; d < hidden; d++ {
		output[d] *= inv
	}
}

// EmbedTo writes a normalized embedding into out (len must equal Dim()).
// The buffer is reused each call; callers should copy if they need to keep the result.
func (m *Model) EmbedTo(text string, out []float32) error {
	if m == nil {
		return errors.New("model is nil")
	}
	if len(out) != m.HiddenSize {
		return fmt.Errorf("output buffer len %d != hidden size %d", len(out), m.HiddenSize)
	}

	tokenIDs, attnMask := m.tokenize(text)
	seqLen := len(tokenIDs)
	m.transformerForward(tokenIDs, seqLen, attnMask)

	meanPooling(out, m.hiddenStates, attnMask, seqLen, m.HiddenSize)
	l2Normalize(out)
	return nil
}

// Embed returns a normalized embedding for the provided text.
func (m *Model) Embed(text string) ([]float32, error) {
	output := make([]float32, m.HiddenSize)
	if err := m.EmbedTo(text, output); err != nil {
		return nil, err
	}
	return output, nil
}

// EmbedBatch embeds multiple texts sequentially and returns a matrix [n][dim].
func (m *Model) EmbedBatch(texts []string) ([][]float32, error) {
	if m == nil {
		return nil, errors.New("model is nil")
	}
	result := make([][]float32, len(texts))
	buf := make([]float32, m.HiddenSize)
	for i, t := range texts {
		if err := m.EmbedTo(t, buf); err != nil {
			return nil, fmt.Errorf("embed %d: %w", i, err)
		}
		copyBuf := make([]float32, m.HiddenSize)
		copy(copyBuf, buf)
		result[i] = copyBuf
	}
	return result, nil
}

// CosineSimilarity returns the cosine similarity for two normalized vectors.
func CosineSimilarity(a, b []float32) (float32, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("dimension mismatch: %d vs %d", len(a), len(b))
	}
	sum := float32(0)
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum, nil
}
