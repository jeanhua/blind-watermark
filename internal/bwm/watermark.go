package bwm

import (
	"encoding/binary"
	"encoding/hex"
	"fmt"
)

// WaterMark provides high-level watermark operations.
type WaterMark struct {
	core       *WaterMarkCore
	passwordWM uint64
	wmBits     []float32
	wmSize     int
}

// NewWaterMark creates a new WaterMark instance.
func NewWaterMark(passwordWM, passwordIMG uint64) *WaterMark {
	p := DefaultParams()
	p.PasswordIMG = passwordIMG
	return &WaterMark{
		core:       NewCore(p),
		passwordWM: passwordWM,
	}
}

// ReadImageFromBytes reads an image from raw BGR float32 data.
func (wm *WaterMark) ReadImageFromBytes(img [][][3]float32) {
	wm.core.ReadImage(img)
}

// WmSize returns the watermark bit count.
func (wm *WaterMark) WmSize() int {
	return wm.wmSize
}

// ReadWatermarkString encodes a string as watermark bits.
func (wm *WaterMark) ReadWatermarkString(content string) {
	encoded := content // Python: content.encode('utf-8')
	hexStr := hex.EncodeToString([]byte(encoded))
	// Convert hex to binary string, then parse
	binLen := len(hexStr) * 4
	bits := make([]float32, binLen)
	hexBytes := []byte(hexStr)
	for i, hb := range hexBytes {
		val := int64(0)
		if hb >= '0' && hb <= '9' {
			val = int64(hb - '0')
		} else if hb >= 'a' && hb <= 'f' {
			val = int64(hb - 'a' + 10)
		} else if hb >= 'A' && hb <= 'F' {
			val = int64(hb - 'A' + 10)
		}
		for b := 0; b < 4; b++ {
			if val&(1<<(3-b)) != 0 {
				bits[i*4+b] = 1
			}
		}
	}
	// Encrypt with password
	wm.wmBits = ShuffleWM(bits, wm.passwordWM)
	wm.wmSize = len(wm.wmBits)
	wm.core.ReadWM(wm.wmBits)
}

// Embed embeds the watermark and returns the watermarked image as BGR float32.
func (wm *WaterMark) Embed() [][][3]float32 {
	return wm.core.Embed()
}

// ExtractRaw extracts raw watermark bits from an image.
func (wm *WaterMark) ExtractRaw(img [][][3]float32) [][]float32 {
	return wm.core.ExtractRaw(img)
}

// ExtractAvg averages raw extraction to get watermark bits.
func (wm *WaterMark) ExtractAvg(raw [][]float32) []float32 {
	return wm.core.ExtractAvg(raw)
}

// ExtractStringFromRaw extracts a string watermark from raw bits using k-means.
func (wm *WaterMark) ExtractStringFromRaw(raw [][]float32, wmLength int) string {
	wm.core.wmSize = wmLength
	avg := wm.core.ExtractAvg(raw)
	// K-means threshold
	decisions := KMeans1D(avg)
	// Convert decisions to bits
	bits := make([]float32, len(decisions))
	for i, d := range decisions {
		if d {
			bits[i] = 1
		}
	}
	// Decrypt
	bits = UnshuffleWM(bits, wm.passwordWM)
	return bitsToString(bits)
}

// bitsToString converts watermark bits back to a string.
func bitsToString(bits []float32) string {
	// Round each bit to 0 or 1
	byteBits := make([]byte, (len(bits)+7)/8)
	for i := 0; i < len(bits); i++ {
		bitVal := 0
		if bits[i] >= 0.5 {
			bitVal = 1
		}
		if bitVal == 1 {
			byteBits[i/8] |= 1 << (7 - i%8)
		}
	}
	// Convert bits string to hex string, then to bytes
	bitStr := ""
	for _, b := range bits {
		if b >= 0.5 {
			bitStr += "1"
		} else {
			bitStr += "0"
		}
	}
	// Pad to multiple of 8
	for len(bitStr)%8 != 0 {
		bitStr = "0" + bitStr
	}
	// Convert to bytes
	hexLen := (len(bitStr) + 3) / 4
	hexBytes := make([]byte, hexLen)
	// Pad bitStr to multiple of 4
	for len(bitStr)%4 != 0 {
		bitStr = "0" + bitStr
	}
	for i := 0; i < hexLen; i++ {
		val := byte(0)
		for b := 0; b < 4; b++ {
			if bitStr[i*4+b] == '1' {
				val |= 1 << (3 - b)
			}
		}
		if val < 10 {
			hexBytes[i] = '0' + val
		} else {
			hexBytes[i] = 'a' + val - 10
		}
	}
	hexStr := string(hexBytes)
	decoded, err := hex.DecodeString(hexStr)
	if err != nil {
		_ = binary.Size(byteBits)
		return fmt.Sprintf("<decode error: %v>", err)
	}
	return string(decoded)
}
