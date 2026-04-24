package gte

import (
	"unsafe"
)

// unsafeString converts a byte slice to a string without allocation.
// The string is only valid while the byte slice is not modified.
func unsafeString(b []byte) string {
	return unsafe.String(unsafe.SliceData(b), len(b))
}

func isPunctuation(b byte) bool {
	return (b >= 33 && b <= 47) || (b >= 58 && b <= 64) || (b >= 91 && b <= 96) || (b >= 123 && b <= 126)
}

func isWhitespace(b byte) bool {
	return b == ' ' || b == '\t' || b == '\n' || b == '\r'
}

func (m *Model) basicTokenize(text string) []string {
	tokens := m.basicBuf[:0]
	i := 0
	for i < len(text) {
		for i < len(text) && isWhitespace(text[i]) {
			i++
		}
		if i >= len(text) {
			break
		}
		start := i
		if isPunctuation(text[i]) {
			i++
		} else {
			for i < len(text) && !isWhitespace(text[i]) && !isPunctuation(text[i]) {
				i++
			}
		}
		src := text[start:i]
		needsLower := false
		for j := 0; j < len(src); j++ {
			if src[j] >= 'A' && src[j] <= 'Z' {
				needsLower = true
				break
			}
		}
		if !needsLower {
			tokens = append(tokens, src)
		} else {
			// Reuse buffer for lowercase
			if cap(m.lowerBuf) < len(src) {
				m.lowerBuf = make([]byte, len(src)*2)
			}
			b := m.lowerBuf[:len(src)]
			for j := 0; j < len(src); j++ {
				c := src[j]
				if c >= 'A' && c <= 'Z' {
					c += 32
				}
				b[j] = c
			}
			// Must copy to string since buffer will be reused
			tokens = append(tokens, string(b))
		}
	}
	m.basicBuf = tokens
	return tokens
}

func (m *Model) wordpieceTokenize(word string, out []int) []int {
	if word == "" {
		return out
	}
	// Reuse a buffer to avoid strings.Builder allocations per subword lookup.
	// Max candidate: "##" + word = 2 + len(word) bytes.
	buf := m.wpBuf[:0]
	start := 0
	for start < len(word) {
		end := len(word)
		found := -1
		for start < end {
			buf = buf[:0]
			if start > 0 {
				buf = append(buf, '#', '#')
			}
			buf = append(buf, word[start:end]...)
			// Lookup without allocating a string: use unsafe conversion.
			candidate := unsafeString(buf)
			if id, ok := m.vocabMap[candidate]; ok {
				found = id
				break
			}
			end--
		}
		if found < 0 {
			out = append(out, tokenUNK)
			start++
		} else {
			out = append(out, found)
			start = end
		}
	}
	m.wpBuf = buf[:0]
	return out
}

func (m *Model) tokenize(text string) ([]int, []bool) {
	basic := m.basicTokenize(text)

	if cap(m.tokenBuf) < m.MaxSeqLen {
		m.tokenBuf = make([]int, 0, m.MaxSeqLen)
	}
	if cap(m.attnMaskBuf) < m.MaxSeqLen {
		m.attnMaskBuf = make([]bool, 0, m.MaxSeqLen)
	}

	tokens := m.tokenBuf[:0]
	attnMask := m.attnMaskBuf[:0]

	tokens = append(tokens, tokenCLS)
	for _, tok := range basic {
		if len(tokens) >= m.MaxSeqLen-1 {
			break
		}
		prevLen := len(tokens)
		tokens = m.wordpieceTokenize(tok, tokens)
		if len(tokens) > m.MaxSeqLen-1 {
			tokens = tokens[:prevLen] // drop token that would overflow
			break
		}
	}
	if len(tokens) < m.MaxSeqLen {
		tokens = append(tokens, tokenSEP)
	}
	attnMask = attnMask[:len(tokens)]
	for i := range attnMask {
		attnMask[i] = true
	}

	m.tokenBuf = tokens
	m.attnMaskBuf = attnMask
	return tokens, attnMask
}

// toLowerNoCopy lowercases ASCII bytes in-place and returns as string without allocation.
func toLowerNoCopy(b []byte) string {
	for i, c := range b {
		if c >= 'A' && c <= 'Z' {
			b[i] = c + 32
		}
	}
	return unsafeString(b)
}
