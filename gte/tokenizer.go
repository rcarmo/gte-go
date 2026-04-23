package gte

import (
	"strings"
)

func isPunctuation(b byte) bool {
	return (b >= 33 && b <= 47) || (b >= 58 && b <= 64) || (b >= 91 && b <= 96) || (b >= 123 && b <= 126)
}

func isWhitespace(b byte) bool {
	return b == ' ' || b == '\t' || b == '\n' || b == '\r'
}

func basicTokenize(text string, buf []string) []string {
	data := []byte(text)
	tokens := buf[:0]
	i := 0
	for i < len(data) {
		for i < len(data) && isWhitespace(data[i]) {
			i++
		}
		if i >= len(data) {
			break
		}
		start := i
		if isPunctuation(data[i]) {
			i++
		} else {
			for i < len(data) && !isWhitespace(data[i]) && !isPunctuation(data[i]) {
				i++
			}
		}
		token := strings.ToLower(string(data[start:i]))
		tokens = append(tokens, token)
	}
	return tokens
}

func (m *Model) wordpieceTokenize(word string, out []int) []int {
	if word == "" {
		return out
	}
	start := 0
	for start < len(word) {
		end := len(word)
		found := -1
		for start < end {
			var b strings.Builder
			if start > 0 {
				b.WriteString("##")
			}
			b.WriteString(word[start:end])
			candidate := b.String()
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
	return out
}

func (m *Model) tokenize(text string) ([]int, []bool) {
	basic := basicTokenize(text, m.basicBuf)

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
