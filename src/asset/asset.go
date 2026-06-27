package asset

import (
	"errors"
	"unsafe"
)

const MagicSGFX uint32 = uint32('S') |
	uint32('G')<<8 |
	uint32('F')<<16 |
	uint32('X')<<24

type Format uint16

const (
	FormatRGB565   Format = 1
	FormatARGB8888 Format = 2
	FormatRGB888   Format = 3
	FormatARGB4444 Format = 4
	FormatARGB1555 Format = 5
	FormatA8       Format = 6
)

var (
	ErrShortAsset    = errors.New("assets: short asset")
	ErrInvalidHeader = errors.New("assets: invalid header")
	ErrInvalidFormat = errors.New("assets: invalid format")
	ErrInvalidData   = errors.New("assets: invalid data")
)

type Asset struct {
	*header
	data unsafe.Pointer
}

func (a Asset) Width() int {
	return int(a.header.width)
}

func (a Asset) Height() int {
	return int(a.header.height)
}

func (a Asset) Format() Format {
	return Format(a.header.format)
}

func (a Asset) Stride() int {
	return int(a.header.stride)
}

func (a Asset) DataLen() int {
	return int(a.header.dataLen)
}

func (a Asset) Data() []byte {
	return unsafe.Slice((*byte)(a.data), int(a.header.dataLen))
}

type header struct {
	magic   uint32 // 'SGFX'
	width   uint16
	height  uint16
	format  Format
	stride  uint16 // bytes per row
	flags   uint32
	dataLen uint32
}

const HeaderSize = 20

func (a *header) validate(availableDataBytes int) error {
	if a.magic != MagicSGFX {
		return ErrInvalidHeader
	}
	if a.width == 0 || a.height == 0 {
		return ErrInvalidHeader
	}

	bpp, ok := BytesPerPixel(a.format)
	if !ok {
		return ErrInvalidFormat
	}

	minStride := uint32(a.width) * uint32(bpp)
	if uint32(a.stride) < minStride {
		return ErrInvalidHeader
	}

	required := uint32(a.stride) * uint32(a.height)
	if a.dataLen < required {
		return ErrInvalidData
	}
	if uint64(a.dataLen) > uint64(availableDataBytes) {
		return ErrInvalidData
	}

	return nil
}

func BytesPerPixel(format Format) (int, bool) {
	switch format {
	case FormatRGB565:
		return 2, true
	case FormatARGB8888:
		return 4, true
	case FormatRGB888:
		return 3, true
	case FormatARGB4444:
		return 2, true
	case FormatARGB1555:
		return 2, true
	case FormatA8:
		return 1, true
	default:
		return 0, false
	}
}
