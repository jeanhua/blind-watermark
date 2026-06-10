package bwm

import "math"

// HaarDWT2 performs a single-level 2D Haar wavelet decomposition.
// Returns CA (approximation) and three detail coefficient matrices: CH, CV, CD.
func HaarDWT2(channel [][]float32) (ca, ch, cv, cd [][]float32) {
	h := len(channel)
	w := len(channel[0])
	// Row decomposition
	rowsHalf := make([][]float32, h)
	for i := 0; i < h; i++ {
		rowsHalf[i] = make([]float32, w)
		for j := 0; j < w/2; j++ {
			a := channel[i][2*j]
			b := channel[i][2*j+1]
			rowsHalf[i][j] = (a + b) / sqrt2     // approximation (low-pass)
			rowsHalf[i][j+w/2] = (a - b) / sqrt2 // detail (high-pass)
		}
	}
	// Column decomposition
	ca = make([][]float32, h/2)
	ch = make([][]float32, h/2)
	cv = make([][]float32, h/2)
	cd = make([][]float32, h/2)
	for i := 0; i < h/2; i++ {
		ca[i] = make([]float32, w/2)
		ch[i] = make([]float32, w/2)
		cv[i] = make([]float32, w/2)
		cd[i] = make([]float32, w/2)
		for j := 0; j < w; j++ {
			a := rowsHalf[2*i][j]
			b := rowsHalf[2*i+1][j]
			if j < w/2 {
				ca[i][j] = (a + b) / sqrt2
				cv[i][j] = (a - b) / sqrt2
			} else {
				ch[i][j-w/2] = (a + b) / sqrt2
				cd[i][j-w/2] = (a - b) / sqrt2
			}
		}
	}
	return
}

// HaarIDWT2 performs a single-level 2D Haar wavelet reconstruction.
func HaarIDWT2(ca, ch, cv, cd [][]float32) [][]float32 {
	h := len(ca)
	w := len(ca[0])
	// Column reconstruction
	rowsFull := make([][]float32, 2*h)
	for i := 0; i < h; i++ {
		rowsFull[2*i] = make([]float32, 2*w)
		rowsFull[2*i+1] = make([]float32, 2*w)
		for j := 0; j < w; j++ {
			// Upper-left (CA)
			a := ca[i][j]
			b := cv[i][j]
			rowsFull[2*i][j] = (a + b) / sqrt2
			rowsFull[2*i+1][j] = (a - b) / sqrt2
			// Right side (CH, CD)
			a = ch[i][j]
			b = cd[i][j]
			rowsFull[2*i][j+w] = (a + b) / sqrt2
			rowsFull[2*i+1][j+w] = (a - b) / sqrt2
		}
	}
	// Row reconstruction
	out := make([][]float32, 2*h)
	for i := 0; i < 2*h; i++ {
		out[i] = make([]float32, 2*w)
		for j := 0; j < w; j++ {
			a := rowsFull[i][j]
			b := rowsFull[i][j+w]
			out[i][2*j] = (a + b) / sqrt2
			out[i][2*j+1] = (a - b) / sqrt2
		}
	}
	return out
}

var sqrt2 = float32(math.Sqrt2)
