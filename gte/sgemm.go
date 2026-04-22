package gte

// sgemm computes C = alpha*A*B^T + beta*C for row-major matrices.
// A is [m, k], B is [n, k] (transposed: B^T is [k, n]), C is [m, n].
// This is a zero-allocation tiled implementation optimized for the matrix
// sizes in GTE-small inference (m=seqLen ≤ ~20, k/n = 384 or 1536).
// For large matrices it falls back to gonum BLAS.
func sgemm(transA, transB bool, m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	// Use zero-alloc serial matmul only for truly small problems.
	// gonum's parallel BLAS is faster for large n*k despite allocation overhead.
	flops := m * n * k
	if flops <= 200000 { // ~200K FMAs — covers QKV (7×1152×384=3M? no that's too big) and attention scores
		if !transA && transB {
			sgemmNT(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
			return
		}
		if !transA && !transB {
			sgemmNN(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
			return
		}
	}
	blasImpl.Sgemm(blasTrans(transA), blasTrans(transB), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

// sgemmNT: C[m,n] = alpha * A[m,k] * B^T[k,n] + beta * C
// B is stored as [n, k] row-major, so B^T access is B[j*ldb + p].
func sgemmNT(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	for i := 0; i < m; i++ {
		aRow := a[i*lda:][:k]
		cRow := c[i*ldc:][:n]
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
			bRow := b[j*ldb:][:k]
			sum := float32(0)
			p := 0
			for ; p+8 <= k; p += 8 {
				sum += aRow[p]*bRow[p] + aRow[p+1]*bRow[p+1] + aRow[p+2]*bRow[p+2] + aRow[p+3]*bRow[p+3]
				sum += aRow[p+4]*bRow[p+4] + aRow[p+5]*bRow[p+5] + aRow[p+6]*bRow[p+6] + aRow[p+7]*bRow[p+7]
			}
			for ; p < k; p++ {
				sum += aRow[p] * bRow[p]
			}
			cRow[j] += alpha * sum
		}
	}
}

// sgemmNN: C[m,n] = alpha * A[m,k] * B[k,n] + beta * C
func sgemmNN(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	for i := 0; i < m; i++ {
		aRow := a[i*lda:][:k]
		cRow := c[i*ldc:][:n]
		if beta == 0 {
			for j := range cRow {
				cRow[j] = 0
			}
		} else if beta != 1 {
			for j := range cRow {
				cRow[j] *= beta
			}
		}
		for p := 0; p < k; p++ {
			aVal := alpha * aRow[p]
			if aVal == 0 {
				continue
			}
			bRow := b[p*ldb:][:n]
			j := 0
			for ; j+8 <= n; j += 8 {
				cRow[j] += aVal * bRow[j]
				cRow[j+1] += aVal * bRow[j+1]
				cRow[j+2] += aVal * bRow[j+2]
				cRow[j+3] += aVal * bRow[j+3]
				cRow[j+4] += aVal * bRow[j+4]
				cRow[j+5] += aVal * bRow[j+5]
				cRow[j+6] += aVal * bRow[j+6]
				cRow[j+7] += aVal * bRow[j+7]
			}
			for ; j < n; j++ {
				cRow[j] += aVal * bRow[j]
			}
		}
	}
}
