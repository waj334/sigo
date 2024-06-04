//go:build samx51 && !generic

package rtc

import (
	"peripheral"
	"runtime/arm/cortexm/sam/chip"
)

const (
	secondNs = 1_000_000_000
)

const (
	OFF Prescaler = iota
	DIV1
	DIV2
	DIV4
	DIV8
	DIV16
	DIV32
	DIV64
	DIV128
	DIV256
	DIV512
	DIV1024
)

const (
	Internal Oscillator = iota
	External
)

const (
	F1K Frequency = iota
	F32K
)

var (
	RTC                  *rtc = &rtc{}
	SOURCE_CLK_FREQUENCY uint32
)

type Prescaler uint8
type Oscillator uint8
type Frequency uint8

type Config struct {
	Prescaler    Prescaler
	ClearOnMatch bool
	Value        uint32
	Compare      [2]uint32
}

type rtc struct {
	prescaler uint16
}

func EnableClocks(oscillator Oscillator, frequency Frequency) error {
	switch oscillator {
	case Internal:
		switch frequency {
		case F1K:
			chip.OSC32KCTRL.RTCCTRL.SetRTCSEL(chip.OSC32KCTRL_RTCCTRL_REG_RTCSEL_ULP1K)
		case F32K:
			chip.OSC32KCTRL.RTCCTRL.SetRTCSEL(chip.OSC32KCTRL_RTCCTRL_REG_RTCSEL_ULP32K)
		}
	case External:
		switch frequency {
		case F1K:
			chip.OSC32KCTRL.RTCCTRL.SetRTCSEL(chip.OSC32KCTRL_RTCCTRL_REG_RTCSEL_XOSC1K)
		case F32K:
			chip.OSC32KCTRL.RTCCTRL.SetRTCSEL(chip.OSC32KCTRL_RTCCTRL_REG_RTCSEL_XOSC32K)
		}
	default:
		return peripheral.ErrInvalidConfig
	}
	chip.MCLK.APBAMASK.SetRTC(true)
	return nil
}

func DisableClocks() {
	chip.MCLK.APBAMASK.SetRTC(false)
}

func (r *rtc) Configure(config Config) error {
	// Disable the RTC before configuring.
	chip.RTC_MODE0.CTRLA.SetENABLE(false)
	for chip.RTC_MODE0.SYNCBUSY.GetENABLE() {
	}

	chip.RTC_MODE0.CTRLA.SetMATCHCLR(config.ClearOnMatch)
	chip.RTC_MODE0.CTRLA.SetMODE(chip.RTC_MODE0_CTRLA_REG_MODE_COUNT32)
	chip.RTC_MODE0.CTRLA.SetCOUNTSYNC(true)
	for chip.RTC_MODE0.SYNCBUSY.GetCOUNTSYNC() {
	}

	// Set the initial compare values.
	r.SetCompareValue(config.Compare)

	// Set the prescaler value.
	switch config.Prescaler {
	case OFF:
		chip.RTC_MODE0.CTRLA.SetPRESCALER(chip.RTC_MODE0_CTRLA_REG_PRESCALER_OFF)
	case DIV1:
		chip.RTC_MODE0.CTRLA.SetPRESCALER(chip.RTC_MODE0_CTRLA_REG_PRESCALER_DIV1)
		r.prescaler = 0
	case DIV2:
		chip.RTC_MODE0.CTRLA.SetPRESCALER(chip.RTC_MODE0_CTRLA_REG_PRESCALER_DIV2)
		r.prescaler = 1
	case DIV4:
		chip.RTC_MODE0.CTRLA.SetPRESCALER(chip.RTC_MODE0_CTRLA_REG_PRESCALER_DIV4)
		r.prescaler = 2
	case DIV8:
		chip.RTC_MODE0.CTRLA.SetPRESCALER(chip.RTC_MODE0_CTRLA_REG_PRESCALER_DIV8)
		r.prescaler = 3
	case DIV16:
		chip.RTC_MODE0.CTRLA.SetPRESCALER(chip.RTC_MODE0_CTRLA_REG_PRESCALER_DIV16)
		r.prescaler = 4
	case DIV32:
		chip.RTC_MODE0.CTRLA.SetPRESCALER(chip.RTC_MODE0_CTRLA_REG_PRESCALER_DIV32)
		r.prescaler = 5
	case DIV64:
		chip.RTC_MODE0.CTRLA.SetPRESCALER(chip.RTC_MODE0_CTRLA_REG_PRESCALER_DIV64)
		r.prescaler = 6
	case DIV128:
		chip.RTC_MODE0.CTRLA.SetPRESCALER(chip.RTC_MODE0_CTRLA_REG_PRESCALER_DIV128)
		r.prescaler = 7
	case DIV256:
		chip.RTC_MODE0.CTRLA.SetPRESCALER(chip.RTC_MODE0_CTRLA_REG_PRESCALER_DIV256)
		r.prescaler = 8
	case DIV512:
		chip.RTC_MODE0.CTRLA.SetPRESCALER(chip.RTC_MODE0_CTRLA_REG_PRESCALER_DIV512)
		r.prescaler = 9
	case DIV1024:
		chip.RTC_MODE0.CTRLA.SetPRESCALER(chip.RTC_MODE0_CTRLA_REG_PRESCALER_DIV1024)
		r.prescaler = 10
	default:
		return peripheral.ErrInvalidConfig
	}

	// Set the initial count value.
	r.SetValue(config.Value)

	// Finally, enable the RTC.
	chip.RTC_MODE0.CTRLA.SetENABLE(true)
	for chip.RTC_MODE0.SYNCBUSY.GetENABLE() {
	}

	return nil
}

func (r *rtc) Value() uint32 {
	for chip.RTC_MODE0.SYNCBUSY.GetCOUNT() {
	}
	return chip.RTC_MODE0.COUNT.GetCOUNT()
}

func (r *rtc) SetValue(value uint32) {
	chip.RTC_MODE0.COUNT.SetCOUNT(value)
	for chip.RTC_MODE0.SYNCBUSY.GetCOUNT() {
	}
}

func (r *rtc) CompareValue() [2]uint32 {
	return [2]uint32{
		chip.RTC_MODE0.COMP[0].GetCOMP(),
		chip.RTC_MODE0.COMP[1].GetCOMP(),
	}
}

func (r *rtc) SetCompareValue(value [2]uint32) {
	chip.RTC_MODE0.COMP[0].SetCOMP(value[0])
	for chip.RTC_MODE0.SYNCBUSY.GetCOMP0() {
	}

	chip.RTC_MODE0.COMP[1].SetCOMP(value[1])
	for chip.RTC_MODE0.SYNCBUSY.GetCOMP1() {
	}
}

func (r *rtc) Frequency() uint32 {
	return SOURCE_CLK_FREQUENCY / (1 << r.prescaler)
}

func (r *rtc) Now() uint64 {
	// Get the current count value.
	count := r.Value()

	// Get the RTC frequency.
	frequency := r.Frequency()

	// Calculate the period of each tick in nanoseconds.
	period := uint64(secondNs) / uint64(frequency)

	// Calculate the elapsed time in nanoseconds.
	return uint64(count) * period
}
