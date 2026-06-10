package imageutil

import (
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"os"
	"path/filepath"
)

// ReadImage reads a JPEG or PNG file and returns BGR float32 pixel data.
func ReadImage(filename string) ([][][3]float32, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	if err != nil {
		return nil, err
	}

	bounds := img.Bounds()
	w := bounds.Dx()
	h := bounds.Dy()

	data := make([][][3]float32, h)
	for y := 0; y < h; y++ {
		data[y] = make([][3]float32, w)
		for x := 0; x < w; x++ {
			r, g, b, _ := img.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()
			data[y][x] = [3]float32{
				float32(b >> 8),
				float32(g >> 8),
				float32(r >> 8),
			}
		}
	}
	return data, nil
}

// WriteImage saves BGR float32 pixel data as a PNG or JPEG file.
func WriteImage(filename string, data [][][3]float32) error {
	h := len(data)
	w := len(data[0])

	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, color.NRGBA{
				R: uint8(clamp(data[y][x][2])),
				G: uint8(clamp(data[y][x][1])),
				B: uint8(clamp(data[y][x][0])),
				A: 255,
			})
		}
	}

	ext := filepath.Ext(filename)
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	switch ext {
	case ".jpg", ".jpeg":
		return jpeg.Encode(f, img, &jpeg.Options{Quality: 100})
	default:
		return png.Encode(f, img)
	}
}

func clamp(v float32) float32 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return v
}
