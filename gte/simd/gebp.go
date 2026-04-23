package simd

import "unsafe"

const gebpNR = 16

var gebpBuf []float32

func ensureGebpBuf(size int) []float32 {
	if cap(gebpBuf) < size {
		gebpBuf = make([]float32, size)
	}
	return gebpBuf[:size]
}

func packBNT(b []float32, ldb, jj, nr, k int, bp []float32) {
	for p := 0; p < k; p++ {
		off := p * gebpNR
		d := 0
		for ; d < nr; d++ {
			bp[off+d] = b[(jj+d)*ldb+p]
		}
		for ; d < gebpNR; d++ {
			bp[off+d] = 0
		}
	}
}

func SgemmNTGebp(m, n, k int, alpha float32, aPtr, bPtr, cPtr unsafe.Pointer, lda, ldb, ldc int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	a := unsafe.Slice((*float32)(aPtr), m*lda)
	b := unsafe.Slice((*float32)(bPtr), n*ldb)
	c := unsafe.Slice((*float32)(cPtr), m*ldc)
	bp := ensureGebpBuf(k * gebpNR)

	for jj := 0; jj < n; jj += gebpNR {
		nr := gebpNR
		if jj+nr > n {
			nr = n - jj
		}
		packBNT(b, ldb, jj, nr, k, bp)

		for ii := 0; ii < m; ii += gebpMR {
			mr := gebpMR
			if ii+mr > m {
				mr = m - ii
			}
			if mr == gebpMR && nr == gebpNR {
				gebpMicroKernel(k, alpha,
					unsafe.Pointer(&a[ii*lda]),
					lda,
					unsafe.Pointer(&bp[0]),
					unsafe.Pointer(&c[ii*ldc+jj]),
					ldc)
			} else {
				for i := 0; i < mr; i++ {
					for d := 0; d < nr; d++ {
						sum := float32(0)
						for p := 0; p < k; p++ {
							sum += a[(ii+i)*lda+p] * bp[p*gebpNR+d]
						}
						c[(ii+i)*ldc+jj+d] += alpha * sum
					}
				}
			}
		}
	}
}
