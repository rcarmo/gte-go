//go:build !amd64 && !arm64

package simd

func Sdot(x, y []float32) float32 {
	sum := float32(0)
	i := 0
	for ; i+8 <= len(x); i += 8 {
		sum += x[i]*y[i] + x[i+1]*y[i+1] + x[i+2]*y[i+2] + x[i+3]*y[i+3] +
			x[i+4]*y[i+4] + x[i+5]*y[i+5] + x[i+6]*y[i+6] + x[i+7]*y[i+7]
	}
	for ; i < len(x); i++ {
		sum += x[i] * y[i]
	}
	return sum
}

func Saxpy(alpha float32, x []float32, y []float32) {
	for i := range x {
		y[i] += alpha * x[i]
	}
}
