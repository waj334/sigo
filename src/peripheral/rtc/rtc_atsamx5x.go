//go:build atsamx5x && !generic

package rtc

import (
	"peripheral"
	"runtime/arm/cortexm/sam/atsamx5x"
	"runtime/arm/cortexm/sam/atsamx5x/support/mclk"
	"runtime/arm/cortexm/sam/atsamx5x/support/osc32kctrl"
	rtc "runtime/arm/cortexm/sam/atsamx5x/support/rtc/mode0"
	"time"
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
	RTC                  *_rtc = &_rtc{}
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
	OnCompare0   func()
	OnCompare1   func()
	OnOverflow   func()
}

type _rtc struct {
	prescaler  uint16
	frequency  uint32
	period     uint32
	onCompare0 func()
	onCompare1 func()
	onOverflow func()
}

func EnableClocks(oscillator Oscillator, frequency Frequency) error {
	switch oscillator {
	case Internal:
		switch frequency {
		case F1K:
			osc32kctrl.Osc32kctrl.Rtcctrl.SetRtcsel(osc32kctrl.RtcctrlRtcselUlp1k)
		case F32K:
			osc32kctrl.Osc32kctrl.Rtcctrl.SetRtcsel(osc32kctrl.RtcctrlRtcselUlp32k)
		}
	case External:
		switch frequency {
		case F1K:
			osc32kctrl.Osc32kctrl.Rtcctrl.SetRtcsel(osc32kctrl.RtcctrlRtcselXosc1k)
		case F32K:
			osc32kctrl.Osc32kctrl.Rtcctrl.SetRtcsel(osc32kctrl.RtcctrlRtcselXosc32k)
		}
	default:
		return peripheral.ErrInvalidConfig
	}
	mclk.Mclk.Apbamask.SetRtc(true)
	return nil
}

func DisableClocks() {
	mclk.Mclk.Apbamask.SetRtc(false)
}

func (r *_rtc) Configure(config Config) error {
	// Disable the RTC before configuring.
	rtc.Mode0.Ctrla.SetEnable(false)
	for rtc.Mode0.Syncbusy.GetEnable() {
	}

	rtc.Mode0.Ctrla.SetMatchclr(config.ClearOnMatch)
	rtc.Mode0.Ctrla.SetMode(rtc.CtrlaModeCount32)
	rtc.Mode0.Ctrla.SetCountsync(true)
	for rtc.Mode0.Syncbusy.GetCountsync() {
	}

	// Set the initial compare values.
	r.SetCompareValue(config.Compare)

	// Set the prescaler value.
	switch config.Prescaler {
	case OFF:
		rtc.Mode0.Ctrla.SetPrescaler(rtc.CtrlaPrescalerOff)
	case DIV1:
		rtc.Mode0.Ctrla.SetPrescaler(rtc.CtrlaPrescalerDiv1)
		r.prescaler = 0
	case DIV2:
		rtc.Mode0.Ctrla.SetPrescaler(rtc.CtrlaPrescalerDiv2)
		r.prescaler = 1
	case DIV4:
		rtc.Mode0.Ctrla.SetPrescaler(rtc.CtrlaPrescalerDiv4)
		r.prescaler = 2
	case DIV8:
		rtc.Mode0.Ctrla.SetPrescaler(rtc.CtrlaPrescalerDiv8)
		r.prescaler = 3
	case DIV16:
		rtc.Mode0.Ctrla.SetPrescaler(rtc.CtrlaPrescalerDiv16)
		r.prescaler = 4
	case DIV32:
		rtc.Mode0.Ctrla.SetPrescaler(rtc.CtrlaPrescalerDiv32)
		r.prescaler = 5
	case DIV64:
		rtc.Mode0.Ctrla.SetPrescaler(rtc.CtrlaPrescalerDiv64)
		r.prescaler = 6
	case DIV128:
		rtc.Mode0.Ctrla.SetPrescaler(rtc.CtrlaPrescalerDiv128)
		r.prescaler = 7
	case DIV256:
		rtc.Mode0.Ctrla.SetPrescaler(rtc.CtrlaPrescalerDiv256)
		r.prescaler = 8
	case DIV512:
		rtc.Mode0.Ctrla.SetPrescaler(rtc.CtrlaPrescalerDiv512)
		r.prescaler = 9
	case DIV1024:
		rtc.Mode0.Ctrla.SetPrescaler(rtc.CtrlaPrescalerDiv1024)
		r.prescaler = 10
	default:
		return peripheral.ErrInvalidConfig
	}

	// Set the initial count value.
	r.SetValue(config.Value)

	// Finally, enable the RTC.
	rtc.Mode0.Ctrla.SetEnable(true)
	for rtc.Mode0.Syncbusy.GetEnable() {
	}

	// Get the RTC frequency.
	r.frequency = SOURCE_CLK_FREQUENCY / (1 << r.prescaler)

	// Calculate the period of each tick in nanoseconds.
	r.period = uint32(time.Second) / r.frequency

	// Enable interrupts.
	r.onCompare0 = config.OnCompare0
	r.onCompare1 = config.OnCompare1
	r.onOverflow = config.OnOverflow

	atsamx5x.IRQ_RTC.EnableIRQ()

	rtc.Mode0.Intenset.SetCmp0(true)
	rtc.Mode0.Intenset.SetCmp1(true)
	rtc.Mode0.Intenset.SetOvf(true)

	return nil
}

func (r *_rtc) Value() uint32 {
	if !rtc.Mode0.Ctrla.GetEnable() {
		return 0
	}

	for rtc.Mode0.Syncbusy.GetCount() {
	}
	return rtc.Mode0.Count.GetCount()
}

func (r *_rtc) SetValue(value uint32) {
	rtc.Mode0.Count.SetCount(value)
	for rtc.Mode0.Syncbusy.GetCount() {
	}
}

func (r *_rtc) CompareValue() [2]uint32 {
	return [2]uint32{
		rtc.Mode0.Comp[0].GetComp(),
		rtc.Mode0.Comp[1].GetComp(),
	}
}

func (r *_rtc) SetCompareValue(value [2]uint32) {
	rtc.Mode0.Comp[0].SetComp(value[0])
	for rtc.Mode0.Syncbusy.GetComp0() {
	}

	rtc.Mode0.Comp[1].SetComp(value[1])
	for rtc.Mode0.Syncbusy.GetComp1() {
	}
}

func (r *_rtc) Frequency() uint32 {
	return r.frequency
}

func (r *_rtc) Period() uint32 {
	return r.period
}

func (r *_rtc) Now() uint64 {
	// Get the current count value.
	count := r.Value()

	// Calculate the elapsed time in nanoseconds.
	return uint64(count) * uint64(r.period)
}

func (r *_rtc) Ticks(d time.Duration) uint32 {
	return uint32(d) / r.period
}

//sigo:interrupt rtcHandler RTCHandler
func rtcHandler() {
	if rtc.Mode0.Intflag.GetCmp0() && RTC.onCompare0 != nil {
		// Call the handler.
		RTC.onCompare0()

		// Clear the interrupt flag.
		rtc.Mode0.Intflag.SetCmp0(true)
	}

	if rtc.Mode0.Intflag.GetCmp1() && RTC.onCompare1 != nil {
		// Call the handler.
		RTC.onCompare1()

		// Clear the interrupt flag.
		rtc.Mode0.Intflag.SetCmp1(true)
	}

	if rtc.Mode0.Intflag.GetOvf() && RTC.onOverflow != nil {
		// Call the handler.
		RTC.onOverflow()

		// Clear the interrupt flag.
		rtc.Mode0.Intflag.SetOvf(true)
	}
}
