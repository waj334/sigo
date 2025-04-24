//go:build generic

package dac

type DAC interface {
	Write(value uint) error
}
