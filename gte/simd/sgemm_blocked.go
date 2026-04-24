package simd

import "unsafe"

// SgemmNTBlockedFMA computes C += alpha * A * B^T with all loops in assembly.
// Single call per sgemm — no Go-side blocking, no allocation overhead.
func SgemmNTBlockedFMA(m, n, k int, alpha float32, aPtr, bPtr, cPtr unsafe.Pointer, lda, ldb, ldc int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	sgemmNTTileFMA(m, n, k, alpha, aPtr, lda, bPtr, ldb, cPtr, ldc)
}
