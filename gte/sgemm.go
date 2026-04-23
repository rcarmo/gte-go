package gte

import "github.com/rcarmo/gte-go/gte/simd"

// sgemm dispatches matrix multiplication to the best available backend.
//
// With CGO_ENABLED=1 and libopenblas-dev installed, this calls OpenBLAS
// directly via CGo — zero Go allocations, AVX2 SIMD, ~7ms/embed.
//
// With CGO_ENABLED=0 (the default), this uses a custom zero-allocation
// implementation with AVX2/FMA assembly for the inner dot product kernel
// (sdot in simd_amd64.s).  Cache-blocked tiling keeps weight tiles in L1.
// This avoids gonum's sgemmParallel goroutine allocations while matching
// or exceeding its throughput for the matrix sizes in GTE-small.
func sgemm(transA, transB bool, m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	if openblasEnabled {
		cblasSgemm(transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
		return
	}
	// Pure Go: gonum's cache-blocked BLAS is hard to beat without a full
	// assembly GEMM tile.  Our AVX2 sdot/saxpy kernels lose to gonum for
	// matrix multiply due to per-call overhead (~15ns × thousands of calls).
	// We use SIMD for non-GEMM operations (layerNorm, gelu, etc.) instead.
	blasImpl.Sgemm(blasTrans(transA), blasTrans(transB), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

const blockSize = 64

// sgemmNTSimd: C = alpha * A * B^T + beta * C, using sdot for inner products.
// A is [m,k], B is [n,k] (row-major), both accessed row-wise.
func sgemmNTSimd(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	for i := 0; i < m; i++ {
		aRow := a[i*lda : i*lda+k]
		cRow := c[i*ldc : i*ldc+n]
		if beta == 0 {
			for j := range cRow {
				cRow[j] = 0
			}
		} else if beta != 1 {
			for j := range cRow {
				cRow[j] *= beta
			}
		}
		for j := 0; j < n; j++ {
			bRow := b[j*ldb : j*ldb+k]
			cRow[j] += alpha * simd.Sdot(aRow, bRow)
		}
	}
}

// sgemmNNSimd: C = alpha * A * B + beta * C, using SIMD saxpy for inner loop.
// A is [m,k], B is [k,n], both row-major.
func sgemmNNSimd(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	for i := 0; i < m; i++ {
		cRow := c[i*ldc : i*ldc+n]
		if beta == 0 {
			for j := range cRow {
				cRow[j] = 0
			}
		} else if beta != 1 {
			for j := range cRow {
				cRow[j] *= beta
			}
		}
	}
	for i := 0; i < m; i++ {
		cRow := c[i*ldc : i*ldc+n]
		for p := 0; p < k; p++ {
			aVal := alpha * a[i*lda+p]
			if aVal == 0 {
				continue
			}
			bRow := b[p*ldb : p*ldb+n]
			simd.Saxpy(aVal, bRow, cRow)
		}
	}
}
