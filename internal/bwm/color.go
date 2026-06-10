package bwm

// BGRToYUV converts a single pixel from BGR to YUV using OpenCV BT.601 coefficients.
// Input channels are 0-255 float32; output YUV is 0-255 float32.
func BGRToYUV(b, g, r float32) (y, u, v float32) {
	y = 0.114*b + 0.587*g + 0.299*r
	u = 0.436*b - 0.28886*g - 0.14713*r + 128
	v = -0.10001*b - 0.51499*g + 0.615*r + 128
	return
}

// YUVToBGR converts a single pixel from YUV back to BGR.
func YUVToBGR(y, u, v float32) (b, g, r float32) {
	r = y + 1.13983*(v-128)
	g = y - 0.39465*(u-128) - 0.58060*(v-128)
	b = y + 2.03211*(u-128)
	return
}

// ImageBGRToYUV converts an entire BGR image (H x W x 3) to YUV float32 planes.
func ImageBGRToYUV(img [][][3]float32) (yPlane, uPlane, vPlane [][]float32) {
	h := len(img)
	w := len(img[0])
	yPlane = make([][]float32, h)
	uPlane = make([][]float32, h)
	vPlane = make([][]float32, h)
	for i := 0; i < h; i++ {
		yPlane[i] = make([]float32, w)
		uPlane[i] = make([]float32, w)
		vPlane[i] = make([]float32, w)
		for j := 0; j < w; j++ {
			yPlane[i][j], uPlane[i][j], vPlane[i][j] = BGRToYUV(img[i][j][0], img[i][j][1], img[i][j][2])
		}
	}
	return
}

// YUVToImageBGR converts YUV float32 planes back to a BGR image.
func YUVToImageBGR(yPlane, uPlane, vPlane [][]float32, h, w int) [][][3]float32 {
	img := make([][][3]float32, h)
	for i := 0; i < h; i++ {
		img[i] = make([][3]float32, w)
		for j := 0; j < w; j++ {
			img[i][j][0], img[i][j][1], img[i][j][2] = YUVToBGR(yPlane[i][j], uPlane[i][j], vPlane[i][j])
		}
	}
	return img
}

// PadToEven adds a white border to make height and width even.
func PadToEven(channel [][]float32) ([][]float32, int, int) {
	h := len(channel)
	w := len(channel[0])
	padH := h % 2
	padW := w % 2
	if padH == 0 && padW == 0 {
		return channel, h, w
	}
	newH := h + padH
	newW := w + padW
	padded := make([][]float32, newH)
	for i := 0; i < newH; i++ {
		padded[i] = make([]float32, newW)
		for j := 0; j < newW; j++ {
			if i < h && j < w {
				padded[i][j] = channel[i][j]
			} else {
				padded[i][j] = 0
			}
		}
	}
	return padded, newH, newW
}

// RemovePad removes the padding, restoring original dimensions.
func RemovePad(channel [][]float32, origH, origW int) [][]float32 {
	return cropChannel(channel, 0, 0, origH, origW)
}

func cropChannel(channel [][]float32, y0, x0, h, w int) [][]float32 {
	out := make([][]float32, h)
	for i := 0; i < h; i++ {
		out[i] = make([]float32, w)
		copy(out[i], channel[y0+i][x0:x0+w])
	}
	return out
}
