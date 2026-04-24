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
	// Process 8 rows at a time for maximum cache utilization.
	// Each B row is contiguous; interleaving 8 rows into the pack buffer
	// maximizes sequential writes.
	d := 0
	for ; d+8 <= nr; d += 8 {
		row0 := b[(jj+d)*ldb:]
		row1 := b[(jj+d+1)*ldb:]
		row2 := b[(jj+d+2)*ldb:]
		row3 := b[(jj+d+3)*ldb:]
		row4 := b[(jj+d+4)*ldb:]
		row5 := b[(jj+d+5)*ldb:]
		row6 := b[(jj+d+6)*ldb:]
		row7 := b[(jj+d+7)*ldb:]
		for p := 0; p < k; p++ {
			off := p*gebpNR + d
			bp[off] = row0[p]
			bp[off+1] = row1[p]
			bp[off+2] = row2[p]
			bp[off+3] = row3[p]
			bp[off+4] = row4[p]
			bp[off+5] = row5[p]
			bp[off+6] = row6[p]
			bp[off+7] = row7[p]
		}
	}
	for ; d+4 <= nr; d += 4 {
		row0 := b[(jj+d)*ldb:]
		row1 := b[(jj+d+1)*ldb:]
		row2 := b[(jj+d+2)*ldb:]
		row3 := b[(jj+d+3)*ldb:]
		for p := 0; p < k; p++ {
			off := p*gebpNR + d
			bp[off] = row0[p]
			bp[off+1] = row1[p]
			bp[off+2] = row2[p]
			bp[off+3] = row3[p]
		}
	}
	for ; d < nr; d++ {
		row := b[(jj+d)*ldb:]
		for p := 0; p < k; p++ {
			bp[p*gebpNR+d] = row[p]
		}
	}
	if nr < gebpNR {
		for p := 0; p < k; p++ {
			for dd := nr; dd < gebpNR; dd++ {
				bp[p*gebpNR+dd] = 0
			}
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
			if nr == gebpNR {
				if mr == gebpMR {
					// Full tile: direct assembly
					gebpMicroKernel(k, alpha,
						unsafe.Pointer(&a[ii*lda]),
						lda,
						unsafe.Pointer(&bp[0]),
						unsafe.Pointer(&c[ii*ldc+jj]),
						ldc)
				} else {
					// Partial m: use temp buffer, call micro-kernel, copy back
					var tmp [gebpMR * gebpNR]float32
					// Copy existing C values for the valid rows
					for i := 0; i < mr; i++ {
						copy(tmp[i*gebpNR:i*gebpNR+gebpNR], c[(ii+i)*ldc+jj:(ii+i)*ldc+jj+gebpNR])
					}
					gebpMicroKernel(k, alpha,
						unsafe.Pointer(&a[ii*lda]),
						lda,
						unsafe.Pointer(&bp[0]),
						unsafe.Pointer(&tmp[0]),
						gebpNR)
					// Copy back only valid rows
					for i := 0; i < mr; i++ {
						copy(c[(ii+i)*ldc+jj:(ii+i)*ldc+jj+gebpNR], tmp[i*gebpNR:i*gebpNR+gebpNR])
					}
				}
			} else {
				// Partial n: scalar
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
