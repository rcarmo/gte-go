package gte

// sgemm dispatches matrix multiplication to the best available backend.
//
// With CGO_ENABLED=1 and libopenblas-dev installed, this calls OpenBLAS
// directly via CGo — zero Go allocations, AVX2 SIMD, ~7ms/embed.
//
// With CGO_ENABLED=0 (the default), this falls back to gonum's pure-Go
// cache-blocked BLAS.  gonum's internal sgemmSerial/sgemmParallel is
// remarkably well-optimized: our hand-rolled blocked kernels couldn't
// match it (150ms vs 14ms).  gonum's only downside is that sgemmParallel
// allocates goroutines per block (~1400 allocs/embed), but the actual
// compute throughput is excellent.  For a zero-dependency static binary
// that runs anywhere, 14ms/embed (5× vs baseline) is a good trade-off.
func sgemm(transA, transB bool, m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	if openblasEnabled {
		cblasSgemm(transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
		return
	}
	// Pure Go: use gonum's cache-blocked BLAS which is well-optimized.
	// Its only downside is goroutine allocations, but for pure-Go builds
	// there's no better alternative without assembly.
	blasImpl.Sgemm(blasTrans(transA), blasTrans(transB), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

const blockSize = 64 // ~16KB per float32 block tile, fits in L1 cache

// sgemmNTBlocked: C += alpha * A * B^T, cache-blocked.
// A is [m,k] row-major, B is [n,k] row-major (transposed access).
// For the NT case, the key insight is that A[i,:] · B[j,:] is a dot product
// along the k dimension. We block on k to keep both A and B tiles in L1.
func sgemmNTBlocked(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, c []float32, ldc int) {
	for i := 0; i < m; i++ {
		aRow := a[i*lda:][:k]
		cRow := c[i*ldc:][:n]
		for jj := 0; jj < n; jj += blockSize {
			jEnd := jj + blockSize
			if jEnd > n {
				jEnd = n
			}
			for j := jj; j < jEnd; j++ {
				bRow := b[j*ldb:][:k]
				sum := float32(0)
				p := 0
				for ; p+8 <= k; p += 8 {
					sum += aRow[p]*bRow[p] + aRow[p+1]*bRow[p+1] +
						aRow[p+2]*bRow[p+2] + aRow[p+3]*bRow[p+3] +
						aRow[p+4]*bRow[p+4] + aRow[p+5]*bRow[p+5] +
						aRow[p+6]*bRow[p+6] + aRow[p+7]*bRow[p+7]
				}
				for ; p < k; p++ {
					sum += aRow[p] * bRow[p]
				}
				cRow[j] += alpha * sum
			}
		}
	}
}

// sgemmNNBlocked: C += alpha * A * B, cache-blocked.
// A is [m,k], B is [k,n], both row-major.
func sgemmNNBlocked(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, c []float32, ldc int) {
	for ii := 0; ii < m; ii += blockSize {
		iEnd := ii + blockSize
		if iEnd > m {
			iEnd = m
		}
		for kk := 0; kk < k; kk += blockSize {
			kEnd := kk + blockSize
			if kEnd > k {
				kEnd = k
			}
			for jj := 0; jj < n; jj += blockSize {
				jEnd := jj + blockSize
				if jEnd > n {
					jEnd = n
				}
				jLen := jEnd - jj
				// Micro-kernel: tile
				for i := ii; i < iEnd; i++ {
					cRow := c[i*ldc+jj:][:jLen]
					for p := kk; p < kEnd; p++ {
						aVal := alpha * a[i*lda+p]
						if aVal == 0 {
							continue
						}
						bRow := b[p*ldb+jj:][:jLen]
						j := 0
						for ; j+8 <= jLen; j += 8 {
							cRow[j] += aVal * bRow[j]
							cRow[j+1] += aVal * bRow[j+1]
							cRow[j+2] += aVal * bRow[j+2]
							cRow[j+3] += aVal * bRow[j+3]
							cRow[j+4] += aVal * bRow[j+4]
							cRow[j+5] += aVal * bRow[j+5]
							cRow[j+6] += aVal * bRow[j+6]
							cRow[j+7] += aVal * bRow[j+7]
						}
						for ; j < jLen; j++ {
							cRow[j] += aVal * bRow[j]
						}
					}
				}
			}
		}
	}
}
