package simd

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

// --- Sdot tests ---

func naiveDot(x, y []float32) float32 {
	sum := float32(0)
	for i := range x {
		sum += x[i] * y[i]
	}
	return sum
}

func TestSdotEmpty(t *testing.T) {
	got := Sdot(nil, nil)
	if got != 0 {
		t.Fatalf("Sdot(nil, nil) = %v, want 0", got)
	}
}

func TestSdotSmall(t *testing.T) {
	for n := 1; n <= 7; n++ {
		x := make([]float32, n)
		y := make([]float32, n)
		for i := range x {
			x[i] = float32(i + 1)
			y[i] = float32(i + 1)
		}
		got := Sdot(x, y)
		want := naiveDot(x, y)
		if relErr(got, want) > 1e-5 {
			t.Errorf("n=%d: Sdot=%v, want=%v", n, got, want)
		}
	}
}

func TestSdotAligned(t *testing.T) {
	for _, n := range []int{8, 16, 32, 64, 128, 384, 512, 1536} {
		x := make([]float32, n)
		y := make([]float32, n)
		rng := rand.New(rand.NewSource(42))
		for i := range x {
			x[i] = rng.Float32()*2 - 1
			y[i] = rng.Float32()*2 - 1
		}
		got := Sdot(x, y)
		want := naiveDot(x, y)
		if relErr(got, want) > 1e-4 {
			t.Errorf("n=%d: Sdot=%v, want=%v, relErr=%v", n, got, want, relErr(got, want))
		}
	}
}

func TestSdotUnaligned(t *testing.T) {
	for _, n := range []int{9, 15, 17, 31, 33, 63, 65, 100, 383, 385} {
		x := make([]float32, n)
		y := make([]float32, n)
		rng := rand.New(rand.NewSource(int64(n)))
		for i := range x {
			x[i] = rng.Float32()*2 - 1
			y[i] = rng.Float32()*2 - 1
		}
		got := Sdot(x, y)
		want := naiveDot(x, y)
		if relErr(got, want) > 1e-4 {
			t.Errorf("n=%d: Sdot=%v, want=%v, relErr=%v", n, got, want, relErr(got, want))
		}
	}
}

// --- Saxpy tests ---

func naiveSaxpy(alpha float32, x, y []float32) {
	for i := range x {
		y[i] += alpha * x[i]
	}
}

func TestSaxpySmall(t *testing.T) {
	for n := 1; n <= 7; n++ {
		x := make([]float32, n)
		y := make([]float32, n)
		ref := make([]float32, n)
		for i := range x {
			x[i] = float32(i + 1)
			y[i] = float32(10 + i)
			ref[i] = y[i]
		}
		Saxpy(2.5, x, y)
		naiveSaxpy(2.5, x, ref)
		for i := range y {
			if relErr(y[i], ref[i]) > 1e-5 {
				t.Errorf("n=%d i=%d: got=%v want=%v", n, i, y[i], ref[i])
			}
		}
	}
}

func TestSaxpyAligned(t *testing.T) {
	for _, n := range []int{8, 16, 32, 64, 128, 384, 1536} {
		x := make([]float32, n)
		y := make([]float32, n)
		ref := make([]float32, n)
		rng := rand.New(rand.NewSource(42))
		for i := range x {
			x[i] = rng.Float32()*2 - 1
			y[i] = rng.Float32()*2 - 1
			ref[i] = y[i]
		}
		Saxpy(0.7, x, y)
		naiveSaxpy(0.7, x, ref)
		for i := range y {
			if relErr(y[i], ref[i]) > 1e-4 {
				t.Errorf("n=%d i=%d: got=%v want=%v relErr=%v", n, i, y[i], ref[i], relErr(y[i], ref[i]))
			}
		}
	}
}


func naiveSgemmNT(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, c []float32, ldc int) {
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := float32(0)
			for p := 0; p < k; p++ {
				sum += a[i*lda+p] * b[j*ldb+p]
			}
			c[i*ldc+j] += alpha * sum
		}
	}
}

func TestSgemmNTIdentity(t *testing.T) {
	if !HasSgemmAsm {
		t.Skip("no SGEMM assembly on this arch")
	}
	// A = I(4), B = I(4), alpha=1 → C += I(4)
	a := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
	b := make([]float32, 16)
	copy(b, a)
	c := make([]float32, 16)

	SgemmNT(4, 4, 4, 1.0, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&c[0]), 4, 4, 4)

	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			want := float32(0)
			if i == j {
				want = 1
			}
			if c[i*4+j] != want {
				t.Errorf("C[%d,%d]=%v, want %v", i, j, c[i*4+j], want)
			}
		}
	}
}

func TestSgemmNTSmall(t *testing.T) {
	if !HasSgemmAsm {
		t.Skip("no SGEMM assembly on this arch")
	}
	m, n, k := 3, 5, 4
	a := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	}
	b := []float32{
		1, 0, 0, 1,
		0, 1, 0, 0,
		0, 0, 1, 0,
		1, 1, 0, 0,
		0, 0, 1, 1,
	}
	alpha := float32(2.0)

	got := make([]float32, m*n)
	ref := make([]float32, m*n)
	SgemmNT(m, n, k, alpha, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&got[0]), k, k, n)
	naiveSgemmNT(m, n, k, alpha, a, k, b, k, ref, n)

	for i := 0; i < m*n; i++ {
		if relErr(got[i], ref[i]) > 1e-5 {
			t.Errorf("C[%d]=%v, want %v", i, got[i], ref[i])
		}
	}
}

func TestSgemmNTGTESizes(t *testing.T) {
	if !HasSgemmAsm {
		t.Skip("no SGEMM assembly on this arch")
	}
	// Test the actual matrix sizes used in GTE-small
	sizes := []struct {
		m, n, k int
		name    string
	}{
		{7, 1152, 384, "QKV fused"},
		{7, 384, 384, "attn output"},
		{7, 1536, 384, "FFN up"},
		{7, 384, 1536, "FFN down"},
		{1, 384, 384, "single token"},
		{16, 384, 384, "longer seq"},
	}

	for _, sz := range sizes {
		t.Run(sz.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(42))
			a := randMatrix(rng, sz.m, sz.k)
			b := randMatrix(rng, sz.n, sz.k)
			got := make([]float32, sz.m*sz.n)
			ref := make([]float32, sz.m*sz.n)
			alpha := float32(0.125)

			SgemmNT(sz.m, sz.n, sz.k, alpha,
				unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&got[0]),
				sz.k, sz.k, sz.n)
			naiveSgemmNT(sz.m, sz.n, sz.k, alpha, a, sz.k, b, sz.k, ref, sz.n)

			maxErr := float32(0)
			for i := range got {
				e := absErr(got[i], ref[i])
				if e > maxErr {
					maxErr = e
				}
			}
			// Allow small floating point divergence for large k
			tol := float32(1e-3)
			if maxErr > tol {
				t.Errorf("maxErr=%v > tol=%v (m=%d n=%d k=%d)", maxErr, tol, sz.m, sz.n, sz.k)
			}
		})
	}
}

func TestSgemmNTWithBeta(t *testing.T) {
	if !HasSgemmAsm {
		t.Skip("no SGEMM assembly on this arch")
	}
	// Verify that += semantics work (C has pre-existing values)
	m, n, k := 3, 8, 16
	rng := rand.New(rand.NewSource(99))
	a := randMatrix(rng, m, k)
	b := randMatrix(rng, n, k)
	alpha := float32(0.5)

	// Pre-fill C with non-zero values
	got := randMatrix(rng, m, n)
	ref := make([]float32, m*n)
	copy(ref, got)

	SgemmNT(m, n, k, alpha, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&got[0]), k, k, n)
	naiveSgemmNT(m, n, k, alpha, a, k, b, k, ref, n)

	for i := range got {
		if relErr(got[i], ref[i]) > 1e-4 {
			t.Errorf("C[%d]=%v, want %v", i, got[i], ref[i])
		}
	}
}

// --- SgemmNN tests ---

func naiveSgemmNN(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, c []float32, ldc int) {
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := float32(0)
			for p := 0; p < k; p++ {
				sum += a[i*lda+p] * b[p*ldb+j]
			}
			c[i*ldc+j] += alpha * sum
		}
	}
}

func TestSgemmNNIdentity(t *testing.T) {
	if !HasSgemmAsm {
		t.Skip("no SGEMM assembly on this arch")
	}
	a := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
	}
	b := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
	c := make([]float32, 8)
	SgemmNN(2, 4, 4, 1.0, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&c[0]), 4, 4, 4)

	for i := range c {
		if c[i] != a[i] {
			t.Errorf("C[%d]=%v, want %v", i, c[i], a[i])
		}
	}
}

func TestSgemmNNSmall(t *testing.T) {
	if !HasSgemmAsm {
		t.Skip("no SGEMM assembly on this arch")
	}
	m, n, k := 2, 3, 4
	a := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
	}
	b := []float32{
		1, 0, 1,
		0, 1, 0,
		1, 0, 1,
		0, 1, 0,
	}
	alpha := float32(1.5)
	got := make([]float32, m*n)
	ref := make([]float32, m*n)

	SgemmNN(m, n, k, alpha, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&got[0]), k, n, n)
	naiveSgemmNN(m, n, k, alpha, a, k, b, n, ref, n)

	for i := range got {
		if relErr(got[i], ref[i]) > 1e-5 {
			t.Errorf("C[%d]=%v, want %v", i, got[i], ref[i])
		}
	}
}

func TestSgemmNNGTESizes(t *testing.T) {
	if !HasSgemmAsm {
		t.Skip("no SGEMM assembly on this arch")
	}
	sizes := []struct {
		m, n, k int
		name    string
	}{
		{7, 32, 7, "attn context (small)"},
		{7, 384, 384, "square"},
		{7, 1536, 384, "FFN-like"},
		{7, 384, 1536, "FFN-like transpose"},
		{1, 384, 7, "single token context"},
		{16, 384, 16, "longer seq context"},
	}

	for _, sz := range sizes {
		t.Run(sz.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(42))
			a := randMatrix(rng, sz.m, sz.k)
			b := randMatrix(rng, sz.k, sz.n)
			got := make([]float32, sz.m*sz.n)
			ref := make([]float32, sz.m*sz.n)
			alpha := float32(0.125)

			SgemmNN(sz.m, sz.n, sz.k, alpha,
				unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&got[0]),
				sz.k, sz.n, sz.n)
			naiveSgemmNN(sz.m, sz.n, sz.k, alpha, a, sz.k, b, sz.n, ref, sz.n)

			maxErr := float32(0)
			for i := range got {
				e := absErr(got[i], ref[i])
				if e > maxErr {
					maxErr = e
				}
			}
			tol := float32(1e-3)
			if maxErr > tol {
				t.Errorf("maxErr=%v > tol=%v (m=%d n=%d k=%d)", maxErr, tol, sz.m, sz.n, sz.k)
			}
		})
	}
}

func TestSgemmNNWithBeta(t *testing.T) {
	if !HasSgemmAsm {
		t.Skip("no SGEMM assembly on this arch")
	}
	m, n, k := 3, 16, 8
	rng := rand.New(rand.NewSource(77))
	a := randMatrix(rng, m, k)
	b := randMatrix(rng, k, n)
	alpha := float32(0.3)

	got := randMatrix(rng, m, n)
	ref := make([]float32, m*n)
	copy(ref, got)

	SgemmNN(m, n, k, alpha, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&got[0]), k, n, n)
	naiveSgemmNN(m, n, k, alpha, a, k, b, n, ref, n)

	for i := range got {
		if relErr(got[i], ref[i]) > 1e-4 {
			t.Errorf("C[%d]=%v, want %v", i, got[i], ref[i])
		}
	}
}

// --- Benchmarks ---

func BenchmarkSdot384(b *testing.B) {
	x := make([]float32, 384)
	y := make([]float32, 384)
	rng := rand.New(rand.NewSource(42))
	for i := range x {
		x[i] = rng.Float32()
		y[i] = rng.Float32()
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Sdot(x, y)
	}
}

func BenchmarkSgemmNT_7x384x384(b *testing.B) {
	if !HasSgemmAsm {
		b.Skip("no SGEMM assembly")
	}
	rng := rand.New(rand.NewSource(42))
	a := randMatrix(rng, 7, 384)
	bm := randMatrix(rng, 384, 384)
	c := make([]float32, 7*384)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SgemmNT(7, 384, 384, 1.0,
			unsafe.Pointer(&a[0]), unsafe.Pointer(&bm[0]), unsafe.Pointer(&c[0]),
			384, 384, 384)
	}
}

func BenchmarkSgemmNT_7x1536x384(b *testing.B) {
	if !HasSgemmAsm {
		b.Skip("no SGEMM assembly")
	}
	rng := rand.New(rand.NewSource(42))
	a := randMatrix(rng, 7, 384)
	bm := randMatrix(rng, 1536, 384)
	c := make([]float32, 7*1536)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SgemmNT(7, 1536, 384, 1.0,
			unsafe.Pointer(&a[0]), unsafe.Pointer(&bm[0]), unsafe.Pointer(&c[0]),
			384, 384, 1536)
	}
}

func BenchmarkSgemmNN_7x384x384(b *testing.B) {
	if !HasSgemmAsm {
		b.Skip("no SGEMM assembly")
	}
	rng := rand.New(rand.NewSource(42))
	a := randMatrix(rng, 7, 384)
	bm := randMatrix(rng, 384, 384)
	c := make([]float32, 7*384)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SgemmNN(7, 384, 384, 1.0,
			unsafe.Pointer(&a[0]), unsafe.Pointer(&bm[0]), unsafe.Pointer(&c[0]),
			384, 384, 384)
	}
}

func BenchmarkSgemmNN_7x1536x384(b *testing.B) {
	if !HasSgemmAsm {
		b.Skip("no SGEMM assembly")
	}
	rng := rand.New(rand.NewSource(42))
	a := randMatrix(rng, 7, 384)
	bm := randMatrix(rng, 384, 1536)
	c := make([]float32, 7*1536)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SgemmNN(7, 1536, 384, 1.0,
			unsafe.Pointer(&a[0]), unsafe.Pointer(&bm[0]), unsafe.Pointer(&c[0]),
			384, 1536, 1536)
	}
}

// --- Helpers ---

func randMatrix(rng *rand.Rand, rows, cols int) []float32 {
	m := make([]float32, rows*cols)
	for i := range m {
		m[i] = rng.Float32()*2 - 1
	}
	return m
}

func relErr(got, want float32) float32 {
	if want == 0 {
		return float32(math.Abs(float64(got)))
	}
	return float32(math.Abs(float64(got-want)) / math.Abs(float64(want)))
}

func absErr(got, want float32) float32 {
	return float32(math.Abs(float64(got - want)))
}

func BenchmarkSgemmNTGebp_7x384x384(b *testing.B) {
	if !HasSgemmAsm {
		b.Skip("no SGEMM assembly")
	}
	rng := rand.New(rand.NewSource(42))
	a := randMatrix(rng, 7, 384)
	bm := randMatrix(rng, 384, 384)
	c := make([]float32, 7*384)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SgemmNTGebp(7, 384, 384, 1.0,
			unsafe.Pointer(&a[0]), unsafe.Pointer(&bm[0]), unsafe.Pointer(&c[0]),
			384, 384, 384)
	}
}

func BenchmarkSgemmNTGebp_7x1536x384(b *testing.B) {
	if !HasSgemmAsm {
		b.Skip("no SGEMM assembly")
	}
	rng := rand.New(rand.NewSource(42))
	a := randMatrix(rng, 7, 384)
	bm := randMatrix(rng, 1536, 384)
	c := make([]float32, 7*1536)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SgemmNTGebp(7, 1536, 384, 1.0,
			unsafe.Pointer(&a[0]), unsafe.Pointer(&bm[0]), unsafe.Pointer(&c[0]),
			384, 384, 1536)
	}
}

// --- SgemmNTGebp tests ---

func TestSgemmNTGebpIdentity(t *testing.T) {
	if !HasSgemmAsm {
		t.Skip("no SGEMM assembly on this arch")
	}
	a := []float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
	b := make([]float32, 16)
	copy(b, a)
	c := make([]float32, 16)
	SgemmNTGebp(4, 4, 4, 1.0, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&c[0]), 4, 4, 4)
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			want := float32(0)
			if i == j {
				want = 1
			}
			if absErr(c[i*4+j], want) > 1e-6 {
				t.Errorf("C[%d,%d]=%v, want %v", i, j, c[i*4+j], want)
			}
		}
	}
}

func TestSgemmNTGebpSmall(t *testing.T) {
	if !HasSgemmAsm {
		t.Skip("no SGEMM assembly")
	}
	m, n, k := 3, 5, 4
	a := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	b := []float32{1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1}
	alpha := float32(2.0)
	got := make([]float32, m*n)
	ref := make([]float32, m*n)
	SgemmNTGebp(m, n, k, alpha, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&got[0]), k, k, n)
	naiveSgemmNT(m, n, k, alpha, a, k, b, k, ref, n)
	for i := range got {
		if relErr(got[i], ref[i]) > 1e-5 {
			t.Errorf("C[%d]=%v, want %v", i, got[i], ref[i])
		}
	}
}

func TestSgemmNTGebpGTESizes(t *testing.T) {
	if !HasSgemmAsm {
		t.Skip("no SGEMM assembly")
	}
	sizes := []struct {
		m, n, k int
		name    string
	}{
		{7, 1152, 384, "QKV fused"},
		{7, 384, 384, "attn output"},
		{7, 1536, 384, "FFN up"},
		{7, 384, 1536, "FFN down"},
		{1, 384, 384, "single token"},
		{16, 384, 384, "longer seq"},
		{64, 384, 384, "large m (uses GEBP path)"},
	}
	for _, sz := range sizes {
		t.Run(sz.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(42))
			a := randMatrix(rng, sz.m, sz.k)
			b := randMatrix(rng, sz.n, sz.k)
			got := make([]float32, sz.m*sz.n)
			ref := make([]float32, sz.m*sz.n)
			alpha := float32(0.125)
			SgemmNTGebp(sz.m, sz.n, sz.k, alpha,
				unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&got[0]),
				sz.k, sz.k, sz.n)
			naiveSgemmNT(sz.m, sz.n, sz.k, alpha, a, sz.k, b, sz.k, ref, sz.n)
			maxErr := float32(0)
			for i := range got {
				e := absErr(got[i], ref[i])
				if e > maxErr {
					maxErr = e
				}
			}
			tol := float32(1e-3)
			if maxErr > tol {
				t.Errorf("maxErr=%v > tol=%v", maxErr, tol)
			}
		})
	}
}

func TestSgemmNTGebpWithPrefilledC(t *testing.T) {
	if !HasSgemmAsm {
		t.Skip("no SGEMM assembly")
	}
	m, n, k := 3, 8, 16
	rng := rand.New(rand.NewSource(99))
	a := randMatrix(rng, m, k)
	b := randMatrix(rng, n, k)
	alpha := float32(0.5)
	got := randMatrix(rng, m, n)
	ref := make([]float32, m*n)
	copy(ref, got)
	SgemmNTGebp(m, n, k, alpha, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&got[0]), k, k, n)
	naiveSgemmNT(m, n, k, alpha, a, k, b, k, ref, n)
	for i := range got {
		if relErr(got[i], ref[i]) > 1e-4 {
			t.Errorf("C[%d]=%v, want %v", i, got[i], ref[i])
		}
	}
}

// --- packBNT tests ---

func TestPackBNT(t *testing.T) {
	// B = [[1,2,3],[4,5,6]]  (2 rows, 3 cols = n=2, k=3)
	b := []float32{1, 2, 3, 4, 5, 6}
	bp := make([]float32, 3*16) // k=3, NR=16
	packBNT(b, 3, 0, 2, 3, bp)

	// bp[p*16 + d] = B[d, p]
	// bp[0*16+0]=B[0,0]=1, bp[0*16+1]=B[1,0]=4
	// bp[1*16+0]=B[0,1]=2, bp[1*16+1]=B[1,1]=5
	// bp[2*16+0]=B[0,2]=3, bp[2*16+1]=B[1,2]=6
	expected := map[int]float32{
		0*16 + 0: 1, 0*16 + 1: 4,
		1*16 + 0: 2, 1*16 + 1: 5,
		2*16 + 0: 3, 2*16 + 1: 6,
	}
	for idx, want := range expected {
		if bp[idx] != want {
			t.Errorf("bp[%d]=%v, want %v", idx, bp[idx], want)
		}
	}
	// Check zero-padding
	for p := 0; p < 3; p++ {
		for d := 2; d < 16; d++ {
			if bp[p*16+d] != 0 {
				t.Errorf("bp[%d*16+%d]=%v, want 0 (padding)", p, d, bp[p*16+d])
			}
		}
	}
}

// --- Saxpy edge cases ---

func TestSaxpyUnaligned(t *testing.T) {
	for _, n := range []int{9, 15, 17, 33, 100, 383} {
		x := make([]float32, n)
		y := make([]float32, n)
		ref := make([]float32, n)
		rng := rand.New(rand.NewSource(int64(n)))
		for i := range x {
			x[i] = rng.Float32()*2 - 1
			y[i] = rng.Float32()*2 - 1
			ref[i] = y[i]
		}
		Saxpy(0.7, x, y)
		naiveSaxpy(0.7, x, ref)
		for i := range y {
			if relErr(y[i], ref[i]) > 1e-4 {
				t.Errorf("n=%d i=%d: got=%v want=%v", n, i, y[i], ref[i])
			}
		}
	}
}

func TestSaxpyZeroAlpha(t *testing.T) {
	x := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	y := []float32{10, 20, 30, 40, 50, 60, 70, 80}
	orig := make([]float32, len(y))
	copy(orig, y)
	Saxpy(0, x, y)
	for i := range y {
		if y[i] != orig[i] {
			t.Errorf("i=%d: y changed with alpha=0: got=%v want=%v", i, y[i], orig[i])
		}
	}
}

// --- SgemmNTBlockedFMA tests ---

func TestSgemmNTBlockedFMAIdentity(t *testing.T) {
	if !HasSgemmAsm {
		t.Skip("no assembly")
	}
	a := []float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
	b := make([]float32, 16)
	copy(b, a)
	c := make([]float32, 16)
	SgemmNTBlockedFMA(4, 4, 4, 1.0, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&c[0]), 4, 4, 4)
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			want := float32(0)
			if i == j {
				want = 1
			}
			if absErr(c[i*4+j], want) > 1e-6 {
				t.Errorf("C[%d,%d]=%v, want %v", i, j, c[i*4+j], want)
			}
		}
	}
}

func TestSgemmNTBlockedFMAGTESizes(t *testing.T) {
	if !HasSgemmAsm {
		t.Skip("no assembly")
	}
	sizes := []struct{ m, n, k int; name string }{
		{7, 1152, 384, "QKV fused"},
		{7, 384, 384, "attn output"},
		{7, 1536, 384, "FFN up"},
		{7, 384, 1536, "FFN down"},
		{1, 384, 384, "single token"},
	}
	for _, sz := range sizes {
		t.Run(sz.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(42))
			a := randMatrix(rng, sz.m, sz.k)
			b := randMatrix(rng, sz.n, sz.k)
			got := make([]float32, sz.m*sz.n)
			ref := make([]float32, sz.m*sz.n)
			alpha := float32(0.125)
			SgemmNTBlockedFMA(sz.m, sz.n, sz.k, alpha,
				unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&got[0]),
				sz.k, sz.k, sz.n)
			naiveSgemmNT(sz.m, sz.n, sz.k, alpha, a, sz.k, b, sz.k, ref, sz.n)
			maxErr := float32(0)
			for i := range got {
				e := absErr(got[i], ref[i])
				if e > maxErr {
					maxErr = e
				}
			}
			if maxErr > 1e-3 {
				t.Errorf("maxErr=%v > 1e-3", maxErr)
			}
		})
	}
}
