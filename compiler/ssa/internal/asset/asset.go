package asset

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"image/draw"
	"os"
	"strings"

	_ "image/jpeg"
	_ "image/png"
)

const (
	magic      = "SGFX"
	HeaderSize = 20
)

type Format uint16

const (
	FormatRGB565 Format = iota + 1
	FormatARGB8888
	FormatRGB888
	FormatARGB4444
	FormatARGB1555
	FormatA8
)

type FormatInfo struct {
	format Format
	bpp    int
}

func (f FormatInfo) Format() Format {
	return f.format
}

func (f FormatInfo) BytesPerPixel() int {
	return f.bpp
}

var formats = map[string]FormatInfo{
	"rgb565":   {FormatRGB565, 2},
	"argb8888": {FormatARGB8888, 4},
	"rgb888":   {FormatRGB888, 3},
	"argb4444": {FormatARGB4444, 2},
	"argb1555": {FormatARGB1555, 2},
	"a8":       {FormatA8, 1},
}

func Info(format string) (FormatInfo, bool) {
	if info, ok := formats[strings.ToLower(format)]; ok {
		return info, true
	}
	return FormatInfo{}, false
}

func Encode(path string, info FormatInfo, strideAlign int, flipY bool) ([]byte, error) {
	src, err := loadImageAsset(path)
	if err != nil {
		return nil, err
	}

	return encodeImageAsset(src, info, strideAlign, flipY)
}

func loadImageAsset(path string) (*image.NRGBA, error) {
	// Open the asset at the specified file path
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open asset file: %w", err)
	}
	defer f.Close()

	// Decode the image.
	src, format, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("failed to decode asset image: %w", err)
	}

	// check if the loaded image is a supported format.
	switch format {
	case "jpeg", "png":
		// Continue
	default:
		return nil, fmt.Errorf("unsupported image format: %s", format)
	}

	// Blit the image onto a fresh buffer.
	bounds := src.Bounds()
	dst := image.NewNRGBA(image.Rect(0, 0, bounds.Dx(), bounds.Dy()))
	draw.Draw(dst, dst.Bounds(), src, bounds.Min, draw.Src)
	return dst, nil
}

func encodeImageAsset(
	src *image.NRGBA,
	info FormatInfo,
	strideAlign int,
	flipY bool,
) ([]byte, error) {
	width := src.Bounds().Dx()
	height := src.Bounds().Dy()

	if width <= 0 || height <= 0 {
		return nil, fmt.Errorf("invalid image dimensions: width=%d, height=%d", width, height)
	}

	if width > 0xFFFF || height > 0xFFFF {
		return nil, fmt.Errorf("image dimensions exceed maximum supported size: width=%d, height=%d", width, height)
	}

	if strideAlign == 0 {
		strideAlign = 4
	}

	stride := alignUp(width*info.bpp, strideAlign)
	if stride > 0xFFFF {
		return nil, fmt.Errorf("stride exceeds maximum supported size: %d", stride)
	}

	// Create the data buffer that will hold the encoded image asset.
	dataLen := stride * height
	buf := bytes.NewBuffer(make([]byte, 0, HeaderSize+dataLen))

	// Format the asset flags.
	var flags uint32
	if flipY {
		flags |= 1
	}

	// Write the asset header.
	buf.Write([]byte(magic))
	_ = binary.Write(buf, binary.LittleEndian, uint16(width))
	_ = binary.Write(buf, binary.LittleEndian, uint16(height))
	_ = binary.Write(buf, binary.LittleEndian, uint16(info.format))
	_ = binary.Write(buf, binary.LittleEndian, uint16(stride))
	_ = binary.Write(buf, binary.LittleEndian, uint32(flags))
	_ = binary.Write(buf, binary.LittleEndian, uint32(dataLen))

	// Write the image data.
	rowPadLen := stride - width*info.bpp
	rowPad := make([]byte, rowPadLen)

	if flipY {
		for y := height - 1; y >= 0; y-- {
			err := writeRow(buf, src, y, width, info.format, rowPad)
			if err != nil {
				return nil, err
			}
		}
	} else {
		for y := 0; y < height; y++ {
			if err := writeRow(buf, src, y, width, info.format, rowPad); err != nil {
				return nil, err
			}
		}
	}

	return buf.Bytes(), nil
}

func writeRow(buf *bytes.Buffer, src *image.NRGBA, y int, width int, format Format, padding []byte) error {
	pix := src.Pix
	for x := 0; x < width; x++ {
		i := src.PixOffset(x, y)
		r, g, b, a := pix[i+0], pix[i+1], pix[i+2], pix[i+3]
		pixel := encodePixel(format, r, g, b, a)
		if _, err := buf.Write(pixel); err != nil {
			return err
		}
	}

	if len(padding) > 0 {
		if _, err := buf.Write(padding); err != nil {
			return err
		}
	}

	return nil
}

func encodePixel(format Format, r, g, b, a byte) []byte {
	switch format {
	case FormatRGB565:
		v := rgb565(r, g, b)
		return []byte{byte(v), byte(v >> 8)}

	case FormatARGB8888:
		// Little-endian 0xAARRGGBB: B, G, R, A.
		return []byte{b, g, r, a}

	case FormatRGB888:
		// Little-endian 0x00RRGGBB-style byte order: B, G, R.
		return []byte{b, g, r}

	case FormatARGB4444:
		v := argb4444(r, g, b, a)
		return []byte{byte(v), byte(v >> 8)}

	case FormatARGB1555:
		v := argb1555(r, g, b, a)
		return []byte{byte(v), byte(v >> 8)}

	case FormatA8:
		return []byte{a}

	default:
		panic("unreachable")
	}
}

func rgb565(r, g, b byte) uint16 {
	return uint16(r&0xF8)<<8 |
		uint16(g&0xFC)<<3 |
		uint16(b)>>3
}

func argb4444(r, g, b, a byte) uint16 {
	return uint16(a>>4)<<12 |
		uint16(r>>4)<<8 |
		uint16(g>>4)<<4 |
		uint16(b>>4)
}

func argb1555(r, g, b, a byte) uint16 {
	var abit uint16
	if a >= 128 {
		abit = 1
	}

	return abit<<15 |
		uint16(r>>3)<<10 |
		uint16(g>>3)<<5 |
		uint16(b>>3)
}

func alignUp(n, align int) int {
	if align <= 1 {
		return n
	}
	return (n + align - 1) &^ (align - 1)
}
