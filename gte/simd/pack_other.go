//go:build !arm64

package simd
const hasNeonPack = false
func packBNTNeon(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15 uintptr, k int, bp uintptr) {
	panic("packBNTNeon: not arm64")
}
