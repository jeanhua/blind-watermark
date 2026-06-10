package bwm

import "math"

// DCT2D performs a 2D DCT on an N x N float32 block (typically 4x4).
func DCT2D(block [][]float32) [][]float32 {
	n := len(block)
	out := make([][]float32, n)
	for i := 0; i < n; i++ {
		out[i] = make([]float32, n)
	}
	rowDCT := make([][]float32, n)
	for i := 0; i < n; i++ {
		rowDCT[i] = dct1D(block[i])
	}
	for j := 0; j < n; j++ {
		col := make([]float32, n)
		for i := 0; i < n; i++ {
			col[i] = rowDCT[i][j]
		}
		colDCT := dct1D(col)
		for i := 0; i < n; i++ {
			out[i][j] = colDCT[i]
		}
	}
	return out
}

// IDCT2D performs a 2D inverse DCT on an N x N float32 block.
func IDCT2D(block [][]float32) [][]float32 {
	n := len(block)
	out := make([][]float32, n)
	for i := 0; i < n; i++ {
		out[i] = make([]float32, n)
	}
	colIDCT := make([][]float32, n)
	for j := 0; j < n; j++ {
		col := make([]float32, n)
		for i := 0; i < n; i++ {
			col[i] = block[i][j]
		}
		colResult := idct1D(col)
		for i := 0; i < n; i++ {
			colIDCT[i] = append(colIDCT[i], colResult[i])
		}
	}
	for i := 0; i < n; i++ {
		out[i] = idct1D(colIDCT[i])
	}
	return out
}

func dct1D(x []float32) []float32 {
	n := len(x)
	out := make([]float32, n)
	piOver2N := math.Pi / (2 * float64(n))
	for k := 0; k < n; k++ {
		var sum float64
		for i := 0; i < n; i++ {
			sum += float64(x[i]) * math.Cos(float64(2*i+1)*float64(k)*piOver2N)
		}
		if k == 0 {
			out[k] = float32(sum * math.Sqrt(1.0/float64(n)))
		} else {
			out[k] = float32(sum * math.Sqrt(2.0/float64(n)))
		}
	}
	return out
}

func idct1D(x []float32) []float32 {
	n := len(x)
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		var sum float64
		piOver2N := math.Pi / (2 * float64(n))
		for k := 0; k < n; k++ {
			factor := 1.0
			if k == 0 {
				factor = 1.0 / math.Sqrt2
			}
			sum += float64(x[k]) * float64(factor) * math.Cos(float64(2*i+1)*float64(k)*piOver2N)
		}
		out[i] = float32(sum * math.Sqrt(2.0/float64(n)))
	}
	return out
}
