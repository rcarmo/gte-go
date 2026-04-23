package simd

import "unsafe"

// SgemmNTBlockedFMA computes C += alpha * A * B^T using cache-blocked tiling
// with FMA-accelerated tile kernels. All inner loops run in assembly.
//
// On amd64, the tile kernel uses VFMADD231PS (1 op) instead of gonum's
// VMULPS+VADDPS (2 ops), giving ~2× inner loop throughput.
//
// For m=7, n=1152, k=384 with bs=64: 108 tile calls, each processing
// 7×64×64 = 28K multiply-adds with FMA.
func SgemmNTBlockedFMA(m, n, k int, alpha float32, aPtr, bPtr, cPtr unsafe.Pointer, lda, ldb, ldc int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	// No k-blocking: process full k in each tile (like gonum's sgemmSerialNotTrans).
	// Only block i and j to improve C-matrix cache locality.
	const bs = 64
	a := (*float32)(aPtr)
	b := (*float32)(bPtr)
	c := (*float32)(cPtr)

	for ii := 0; ii < m; ii += bs {
		iLen := bs
		if ii+iLen > m {
			iLen = m - ii
		}
		for jj := 0; jj < n; jj += bs {
			jLen := bs
			if jj+jLen > n {
				jLen = n - jj
			}
			aOff := unsafe.Pointer(unsafe.Add(unsafe.Pointer(a), 4*(ii*lda)))
			bOff := unsafe.Pointer(unsafe.Add(unsafe.Pointer(b), 4*(jj*ldb)))
			cOff := unsafe.Pointer(unsafe.Add(unsafe.Pointer(c), 4*(ii*ldc+jj)))
			sgemmNTTileFMA(iLen, jLen, k, alpha, aOff, lda, bOff, ldb, cOff, ldc)
		}
	}
}
