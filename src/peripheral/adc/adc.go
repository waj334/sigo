//go:build generic

package adc

type ADC interface {
	Read(channel int) (uint, error)
}
