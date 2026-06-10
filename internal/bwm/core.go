package bwm

import (
	"math"
	"math/rand/v2"
	"sync"
)

// Params holds watermark algorithm parameters.
type Params struct {
	PasswordIMG uint64
	D1, D2      float32
	BlockShape  [2]int
}

// DefaultParams returns default watermark parameters.
func DefaultParams() Params {
	return Params{
		PasswordIMG: 1,
		D1:          36,
		D2:          20,
		BlockShape:  [2]int{4, 4},
	}
}

// WaterMarkCore holds the state for embedding/extracting watermarks.
type WaterMarkCore struct {
	Params
	img  [][][3]float32
	imgH int
	imgW int
	// YUV planes (padded to even)
	yPlane, uPlane, vPlane [][]float32
	yOrigH, yOrigW         int
	// DWT results per channel
	caY, caU, caV [][]float32
	chY, cvY, cdY [][]float32
	chU, cvU, cdU [][]float32
	chV, cvV, cdV [][]float32
	// Block decomposition
	caBlockY, caBlockU, caBlockV [][][][]float32
	caBlockRows, caBlockCols     int
	// Watermark data
	wmBits []float32
	wmSize int
	// Shuffle indices
	shuffleIdx [][]int
	blockNum   int
	blockSize  int
	partH      int
	partW      int
}

// NewCore creates a new WaterMarkCore with given params.
func NewCore(p Params) *WaterMarkCore {
	if p.BlockShape[0] == 0 {
		p.BlockShape = [2]int{4, 4}
	}
	if p.D1 == 0 {
		p.D1 = 36
	}
	return &WaterMarkCore{Params: p}
}

// ReadImage reads image from BGR float32 data.
func (c *WaterMarkCore) ReadImage(img [][][3]float32) {
	c.img = img
	c.imgH = len(img)
	c.imgW = len(img[0])
	c.yPlane, c.uPlane, c.vPlane = ImageBGRToYUVParallel(img)
	c.yOrigH, c.yOrigW = c.imgH, c.imgW
	// Pad to even
	c.yPlane, _, _ = PadToEven(c.yPlane)
	c.uPlane, _, _ = PadToEven(c.uPlane)
	c.vPlane, _, _ = PadToEven(c.vPlane)
	// DWT on each channel (concurrent)
	var wg sync.WaitGroup
	wg.Add(3)
	go func() { defer wg.Done(); c.caY, c.chY, c.cvY, c.cdY = HaarDWT2(c.yPlane) }()
	go func() { defer wg.Done(); c.caU, c.chU, c.cvU, c.cdU = HaarDWT2(c.uPlane) }()
	go func() { defer wg.Done(); c.caV, c.chV, c.cvV, c.cdV = HaarDWT2(c.vPlane) }()
	wg.Wait()
	// Decompose CA into blocks
	c.caBlockY = blockSplit(c.caY, c.BlockShape[0], c.BlockShape[1])
	c.caBlockU = blockSplit(c.caU, c.BlockShape[0], c.BlockShape[1])
	c.caBlockV = blockSplit(c.caV, c.BlockShape[0], c.BlockShape[1])
	if len(c.caBlockY) > 0 {
		c.caBlockRows = len(c.caBlockY)
		c.caBlockCols = len(c.caBlockY[0])
	}
	c.blockNum = c.caBlockRows * c.caBlockCols
	c.blockSize = c.BlockShape[0] * c.BlockShape[1]
	c.partH = c.caBlockRows * c.BlockShape[0]
	c.partW = c.caBlockCols * c.BlockShape[1]
}

// ReadWM sets the watermark bits.
func (c *WaterMarkCore) ReadWM(bits []float32) {
	c.wmBits = bits
	c.wmSize = len(bits)
	if c.blockNum == 0 {
		return
	}
	if c.wmSize > c.blockNum {
		panic("watermark too large for image")
	}
	c.shuffleIdx = ShuffleStrategy(c.PasswordIMG, c.blockNum, c.blockSize)
}

// Embed embeds the watermark and returns the watermarked BGR image.
// Channel-level and block-level processing run concurrently.
func (c *WaterMarkCore) Embed() [][][3]float32 {
	if c.shuffleIdx == nil {
		c.shuffleIdx = ShuffleStrategy(c.PasswordIMG, c.blockNum, c.blockSize)
	}

	// Embed on each channel: Y, U, V run concurrently
	var wg sync.WaitGroup
	wg.Add(3)
	go func() { defer wg.Done(); c.processChannelEmbed(c.caBlockY, c.shuffleIdx, c.caY) }()
	go func() { defer wg.Done(); c.processChannelEmbed(c.caBlockU, c.shuffleIdx, c.caU) }()
	go func() { defer wg.Done(); c.processChannelEmbed(c.caBlockV, c.shuffleIdx, c.caV) }()
	wg.Wait()

	// IDWT on each channel concurrently
	var yRec, uRec, vRec [][]float32
	wg.Add(3)
	go func() { defer wg.Done(); yRec = HaarIDWT2(c.caY, c.chY, c.cvY, c.cdY) }()
	go func() { defer wg.Done(); uRec = HaarIDWT2(c.caU, c.chU, c.cvU, c.cdU) }()
	go func() { defer wg.Done(); vRec = HaarIDWT2(c.caV, c.chV, c.cvV, c.cdV) }()
	wg.Wait()

	// Remove padding
	yRec = RemovePad(yRec, c.yOrigH, c.yOrigW)
	uRec = RemovePad(uRec, c.yOrigH, c.yOrigW)
	vRec = RemovePad(vRec, c.yOrigH, c.yOrigW)

	// YUV -> BGR (parallel row conversion)
	result := YUVToImageBGRParallel(yRec, uRec, vRec, c.yOrigH, c.yOrigW)

	// Clip to 0-255 (parallel by row)
	clipParallel(result)
	return result
}

// processChannelEmbed embeds watermark into one channel's blocks, using concurrent block processing.
func (c *WaterMarkCore) processChannelEmbed(caBlock [][][][]float32, shuffleIdx [][]int, ca [][]float32) {
	rows := len(caBlock)
	cols := len(caBlock[0])
	total := rows * cols

	// Process blocks concurrently. Each goroutine handles a stripe of rows.
	var wg sync.WaitGroup
	for bi := 0; bi < rows; bi++ {
		wg.Add(1)
		go func(ri int) {
			defer wg.Done()
			for cj := 0; cj < cols; cj++ {
				idx := ri*cols + cj
				block := caBlock[ri][cj]
				caBlock[ri][cj] = blockEmbed(block, shuffleIdx[idx], c.wmBits[idx%c.wmSize], c.D1, c.D2)
			}
		}(bi)
	}
	wg.Wait()

	// Reassemble blocks into CA
	caPart := blockJoin(caBlock)
	for i := 0; i < c.partH; i++ {
		for j := 0; j < c.partW; j++ {
			ca[i][j] = caPart[i][j]
		}
	}
	_ = total
}

func blockEmbed(block [][]float32, shuffle []int, wmBit float32, d1, d2 float32) [][]float32 {
	blockDCT := DCT2D(block)
	// Flatten and shuffle
	flat := flatten(blockDCT, len(shuffle))
	shuffled := ShuffleArray(flat, shuffle)
	shuffledMat := unflatten(shuffled, len(block), len(block[0]))
	// SVD
	u, s, v := JacobiSVD(shuffledMat)
	// Modify singular values
	s[0] = float32(math.Floor(float64(s[0]/d1))+0.25+0.5*float64(wmBit)) * d1
	if d2 > 0 {
		s[1] = float32(math.Floor(float64(s[1]/d2))+0.25+0.5*float64(wmBit)) * d2
	}
	// Reconstruct
	recon := DiagMatMul(u, s, v)
	reconFlat := flatten(recon, len(shuffle))
	unshuffled := UnshuffleArray(reconFlat, shuffle)
	return IDCT2D(unflatten(unshuffled, len(block), len(block[0])))
}

// flatten converts an r x c matrix to a flat slice of length total.
func flatten(mat [][]float32, total int) []float32 {
	out := make([]float32, total)
	c := len(mat[0])
	for i := 0; i < len(mat); i++ {
		copy(out[i*c:(i+1)*c], mat[i])
	}
	return out
}

// unflatten converts a flat slice to an r x c matrix.
func unflatten(flat []float32, r, c int) [][]float32 {
	out := make([][]float32, r)
	for i := 0; i < r; i++ {
		out[i] = make([]float32, c)
		copy(out[i], flat[i*c:(i+1)*c])
	}
	return out
}

// ExtractRaw extracts raw watermark bits from each block and channel (concurrent channels).
func (c *WaterMarkCore) ExtractRaw(img [][][3]float32) [][]float32 {
	c.ReadImage(img)
	if c.shuffleIdx == nil {
		c.shuffleIdx = ShuffleStrategy(c.PasswordIMG, c.blockNum, c.blockSize)
	}
	raw := make([][]float32, 3)
	var wg sync.WaitGroup
	wg.Add(3)
	go func() {
		defer wg.Done()
		raw[0] = c.extractChannel(c.caBlockY, c.caBlockRows, c.caBlockCols, c.shuffleIdx)
	}()
	go func() {
		defer wg.Done()
		raw[1] = c.extractChannel(c.caBlockU, c.caBlockRows, c.caBlockCols, c.shuffleIdx)
	}()
	go func() {
		defer wg.Done()
		raw[2] = c.extractChannel(c.caBlockV, c.caBlockRows, c.caBlockCols, c.shuffleIdx)
	}()
	wg.Wait()
	return raw
}

func (c *WaterMarkCore) extractChannel(caBlock [][][][]float32, rows, cols int, shuffleIdx [][]int) []float32 {
	total := rows * cols
	result := make([]float32, total)
	// Concurrent block extraction per row stripe
	var wg sync.WaitGroup
	for bi := 0; bi < rows; bi++ {
		wg.Add(1)
		go func(ri int) {
			defer wg.Done()
			for cj := 0; cj < cols; cj++ {
				idx := ri*cols + cj
				result[idx] = blockExtract(caBlock[ri][cj], shuffleIdx[idx], c.D1, c.D2)
			}
		}(bi)
	}
	wg.Wait()
	return result
}

func blockExtract(block [][]float32, shuffle []int, d1, d2 float32) float32 {
	blockDCT := DCT2D(block)
	flat := flatten(blockDCT, len(shuffle))
	shuffled := ShuffleArray(flat, shuffle)
	shuffledMat := unflatten(shuffled, len(block), len(block[0]))
	_, s, _ := JacobiSVD(shuffledMat)
	wm := float32(0)
	if math.Mod(float64(s[0]), float64(d1)) > float64(d1)/2 {
		wm = 1
	}
	if d2 > 0 {
		tmp := float32(0)
		if math.Mod(float64(s[1]), float64(d2)) > float64(d2)/2 {
			tmp = 1
		}
		wm = (wm*3 + tmp*1) / 4
	}
	return wm
}

// ExtractAvg averages watermark bits across channels and repeated blocks.
func (c *WaterMarkCore) ExtractAvg(raw [][]float32) []float32 {
	avg := make([]float32, c.wmSize)
	for i := 0; i < c.wmSize; i++ {
		var sum float32
		count := 0
		for ch := 0; ch < 3; ch++ {
			for j := i; j < len(raw[ch]); j += c.wmSize {
				sum += raw[ch][j]
				count++
			}
		}
		if count > 0 {
			avg[i] = sum / float32(count)
		}
	}
	return avg
}

// KMeans1D performs 1D k-means thresholding on input data.
func KMeans1D(inputs []float32) []bool {
	minVal := inputs[0]
	maxVal := inputs[0]
	for _, v := range inputs {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}
	center := [2]float32{minVal, maxVal}
	eTol := float32(1e-6)
	var threshold float32
	for iter := 0; iter < 300; iter++ {
		threshold = (center[0] + center[1]) / 2
		var sum0, sum1 float32
		count0, count1 := 0, 0
		for _, v := range inputs {
			if v > threshold {
				sum1 += v
				count1++
			} else {
				sum0 += v
				count0++
			}
		}
		if count0 > 0 {
			center[0] = sum0 / float32(count0)
		}
		if count1 > 0 {
			center[1] = sum1 / float32(count1)
		}
		newThresh := (center[0] + center[1]) / 2
		if absF32(newThresh-threshold) < eTol {
			threshold = newThresh
			break
		}
	}
	result := make([]bool, len(inputs))
	for i, v := range inputs {
		result[i] = v > threshold
	}
	return result
}

func absF32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

// blockSplit splits a 2D array into 4D blocks.
func blockSplit(mat [][]float32, bh, bw int) [][][][]float32 {
	h := len(mat)
	w := len(mat[0])
	rows := h / bh
	cols := w / bw
	result := make([][][][]float32, rows)
	for i := 0; i < rows; i++ {
		result[i] = make([][][]float32, cols)
		for j := 0; j < cols; j++ {
			result[i][j] = make([][]float32, bh)
			for bi := 0; bi < bh; bi++ {
				result[i][j][bi] = make([]float32, bw)
				copy(result[i][j][bi], mat[i*bh+bi][j*bw:(j+1)*bw])
			}
		}
	}
	return result
}

// blockJoin joins a 4D block array back into a 2D matrix.
func blockJoin(blocks [][][][]float32) [][]float32 {
	rows := len(blocks)
	cols := len(blocks[0])
	bh := len(blocks[0][0])
	bw := len(blocks[0][0][0])
	h := rows * bh
	w := cols * bw
	result := make([][]float32, h)
	for i := 0; i < h; i++ {
		result[i] = make([]float32, w)
	}
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			for bi := 0; bi < bh; bi++ {
				copy(result[i*bh+bi][j*bw:(j+1)*bw], blocks[i][j][bi])
			}
		}
	}
	return result
}

// ShuffleWM encrypts watermark bits using password.
func ShuffleWM(bits []float32, seed uint64) []float32 {
	n := len(bits)
	perm := make([]int, n)
	for i := 0; i < n; i++ {
		perm[i] = i
	}
	rng := rand.New(rand.NewPCG(seed, seed))
	rng.Shuffle(n, func(i, j int) {
		perm[i], perm[j] = perm[j], perm[i]
	})
	result := make([]float32, n)
	for i, p := range perm {
		result[i] = bits[p]
	}
	return result
}

// UnshuffleWM reverses watermark shuffle.
func UnshuffleWM(bits []float32, seed uint64) []float32 {
	n := len(bits)
	perm := make([]int, n)
	for i := 0; i < n; i++ {
		perm[i] = i
	}
	rng := rand.New(rand.NewPCG(seed, seed))
	rng.Shuffle(n, func(i, j int) {
		perm[i], perm[j] = perm[j], perm[i]
	})
	inverse := make([]int, n)
	for i, p := range perm {
		inverse[p] = i
	}
	result := make([]float32, n)
	for i, p := range inverse {
		result[i] = bits[p]
	}
	return result
}

// clipParallel clips BGR pixels to [0,255] in parallel.
func clipParallel(img [][][3]float32) {
	h := len(img)
	var wg sync.WaitGroup
	for i := 0; i < h; i++ {
		wg.Add(1)
		go func(row int) {
			defer wg.Done()
			for j := range img[row] {
				for k := 0; k < 3; k++ {
					img[row][j][k] = clamp(img[row][j][k], 0, 255)
				}
			}
		}(i)
	}
	wg.Wait()
}

func clamp(v, lo, hi float32) float32 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
