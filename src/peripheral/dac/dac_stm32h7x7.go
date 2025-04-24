//go:build stm32h7x7

package dac

import (
	"peripheral"
	"runtime/arm/cortexm/stm32/stm32h7x7/support/dac"
	"sync"
)

const (
	DAC1 _dac = iota
	DAC2
)

var (
	mutex      [2]sync.Mutex
	resolution = [2]Resolution{
		Resolution12Bit,
		Resolution12Bit,
	}
)

type Resolution uint8

const (
	Resolution8Bit Resolution = iota
	Resolution12Bit
	Resolution12BitL
)

type Config struct {
	Enable     bool
	Resolution Resolution
}

type _dac uint8

func (d _dac) Configure(config Config) error {
	// Validate the configuration.
	if config.Resolution > Resolution12BitL {
		return peripheral.ErrInvalidConfig
	}

	if d > DAC2 {
		return peripheral.ErrInvalidPinout
	}

	mutex[d].Lock()

	// Enable DAC channel.
	switch d {
	case DAC1:
		dac.Dac.Cr.SetEn1(config.Enable)
	case DAC2:
		dac.Dac.Cr.SetEn2(config.Enable)
	}

	// Set the resolution for this DAC channel.
	resolution[d] = config.Resolution

	mutex[d].Unlock()
	return nil
}

func (d _dac) Write(value uint) error {
	mutex[d].Lock()
	switch d {
	case DAC1:
		if !dac.Dac.Cr.GetEn1() {
			mutex[d].Unlock()
			return peripheral.ErrInvalidState
		}
		switch resolution[d] {
		case Resolution8Bit:
			dac.Dac.Dhr8r1.SetDacc1dhr(uint8(value))
		case Resolution12Bit:
			dac.Dac.Dhr12r1.SetDacc1dhr(uint16(value & 0xFFF))
		case Resolution12BitL:
			dac.Dac.Dhr12l1.SetDacc1dhr(uint16(value&0xFFF) << 4)
		default:
			panic("unreachable")
		}
	case DAC2:
		if !dac.Dac.Cr.GetEn2() {
			mutex[d].Unlock()
			return peripheral.ErrInvalidState
		}
		switch resolution[d] {
		case Resolution8Bit:
			dac.Dac.Dhr8r2.SetDacc2dhr(uint8(value))
		case Resolution12Bit:
			dac.Dac.Dhr12r2.SetDacc2dhr(uint16(value & 0xFFF))
		case Resolution12BitL:
			dac.Dac.Dhr12l2.SetDacc2dhr(uint16(value&0xFFF) << 4)
		default:
			panic("unreachable")
		}
	}
	mutex[d].Unlock()
	return nil
}
