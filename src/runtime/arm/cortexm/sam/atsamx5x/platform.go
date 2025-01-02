//go:build atsamx5x

package atsamx5x

import (
	"runtime/arm/cortexm"
	"runtime/arm/cortexm/sam/atsamx5x/support/gclk"
	"runtime/arm/cortexm/sam/atsamx5x/support/mclk"
	"runtime/arm/cortexm/sam/atsamx5x/support/osc32kctrl"
	"runtime/arm/cortexm/sam/atsamx5x/support/oscctrl"
	"runtime/arm/cortexm/sam/atsamx5x/support/systemcontrol"
)

const (
	GCLK0 = iota
	GCLK1
	GCLK2
	GCLK3
	GCLK4
	GCLK5
	GCLK6
	GCLK7

	GCLK_OSCCTRL_DFLL48     = 0
	GCLK_OSCCTRL_FDPLL0     = 1
	GCLK_OSCCTRL_FDPLL1     = 2
	GCLK_OSCCTRL_FDPLL0_32K = 3
	GCLK_OSCCTRL_FDPLL1_32K
	GCLK_SDHC0_SLOW
	GCLK_SDHC1_SLOW
	GCLK_SERCOM0_SLOW
	GCLK_SERCOM1_SLOW
	GCLK_SERCOM2_SLOW
	GCLK_SERCOM3_SLOW
	GCLK_SERCOM4_SLOW
	GCLK_SERCOM5_SLOW
	GCLK_SERCOM6_SLOW
	GCLK_SERCOM7_SLOW
	GCLK_EIC          = 4
	GCLK_FREQM_MSR    = 5
	GCLK_FREQM_REF    = 6
	GCLK_SERCOM0_CORE = 7
	GCLK_SERCOM1_CORE = 8
	GCLK_TC0          = 9
	GCLK_TC1
	GCLK_USB          = 10
	GCLK_EVSYS0       = 11
	GCLK_EVSYS1       = 12
	GCLK_EVSYS2       = 13
	GCLK_EVSYS3       = 14
	GCLK_EVSYS4       = 15
	GCLK_EVSYS5       = 16
	GCLK_EVSYS6       = 17
	GCLK_EVSYS7       = 18
	GCLK_EVSYS8       = 19
	GCLK_EVSYS9       = 20
	GCLK_EVSYS10      = 21
	GCLK_EVSYS11      = 22
	GCLK_SERCOM2_CORE = 23
	GCLK_SERCOM3_CORE = 24
	GCLK_TCC0         = 25
	GCLK_TCC1
	GCLK_TC2 = 26
	GCLK_TC3
	GCLK_CAN0 = 27
	GCLK_CAN1 = 28
	GCLK_TCC2 = 29
	GCLK_TCC3
	GCLK_TC4 = 30
	GCLK_TC5
	GCLK_PDEC         = 31
	GCLK_AC           = 32
	GCLK_CCL          = 33
	GCLK_SERCOM4_CORE = 34
	GCLK_SERCOM5_CORE = 35
	GCLK_SERCOM6_CORE = 36
	GCLK_SERCOM7_CORE = 37
	GCLK_TCC4         = 38
	GCLK_TC6          = 39
	GCLK_TC7
	GCLK_ADC0      = 40
	GCLK_ADC1      = 41
	GCLK_DAC       = 42
	GCLK_I2S0      = 43
	GCLK_I2S1      = 44
	GCLK_SDHC0     = 45
	GCLK_SDHC1     = 46
	GCLK_CM4_TRACE = 47
)

var (
	SERCOM_REF_FREQUENCY uint32 = 60_000_000
	GCLK0_FREQUENCY      uint32 = 120_000_000
)

func init() {
	cortexm.SYSTICK_FREQUENCY = GCLK0_FREQUENCY
	cortexm.NPRIORITY_BITS = 3
}

func DefaultClocks() {
	// Configure the XOSC32K oscillator
	osc32kctrl.Osc32kctrl.Xosc32k.SetEnable(false)
	osc32kctrl.Osc32kctrl.Xosc32k.SetCgm(osc32kctrl.Xosc32kCgmXt)
	osc32kctrl.Osc32kctrl.Xosc32k.SetXtalen(true)
	osc32kctrl.Osc32kctrl.Xosc32k.SetEn32k(true)
	osc32kctrl.Osc32kctrl.Xosc32k.SetOndemand(false)
	osc32kctrl.Osc32kctrl.Xosc32k.SetRunstdby(true)
	osc32kctrl.Osc32kctrl.Xosc32k.SetStartup(osc32kctrl.Xosc32kStartupCycle2048)
	osc32kctrl.Osc32kctrl.Xosc32k.SetEnable(true)

	// Wait for XOSC32K to be stable
	for !osc32kctrl.Osc32kctrl.Intflag.GetXosc32krdy() {
	}

	// Set RTC clock
	osc32kctrl.Osc32kctrl.Rtcctrl.SetRtcsel(osc32kctrl.RtcctrlRtcselXosc32k)

	// Enable DFLL - 48MHz
	oscctrl.Oscctrl.Dpll[0].Dpllctrla.SetOndemand(false)
	for oscctrl.Oscctrl.Dpll[0].Dpllsyncbusy.GetEnable() {
	}

	oscctrl.Oscctrl.Dpll[0].Dpllctrla.SetEnable(true)
	for oscctrl.Oscctrl.Dpll[0].Dpllsyncbusy.GetEnable() {
	}

	// Set up GCLK2 - 1MHz
	gclk.Gclk.Genctrl[GCLK2].SetDiv(48)
	for gclk.Gclk.Syncbusy.GetGenctrl()&gclk.SyncbusyGenctrlGclk2 != 0 {
	}

	gclk.Gclk.Genctrl[GCLK2].SetSrc(gclk.GenctrlSrcDfll)
	for gclk.Gclk.Syncbusy.GetGenctrl()&gclk.SyncbusyGenctrlGclk2 != 0 {
	}

	gclk.Gclk.Genctrl[GCLK2].SetGenen(true)
	for gclk.Gclk.Syncbusy.GetGenctrl()&gclk.SyncbusyGenctrlGclk2 != 0 {
	}

	// Configure the DPLL - 120MHz
	gclk.Gclk.Pchctrl[GCLK_OSCCTRL_FDPLL0].SetChen(false)
	for gclk.Gclk.Pchctrl[GCLK_OSCCTRL_FDPLL0].GetChen() {
	}

	gclk.Gclk.Pchctrl[GCLK_OSCCTRL_FDPLL0].SetGen(gclk.PchctrlGenGclk2)
	gclk.Gclk.Pchctrl[GCLK_OSCCTRL_FDPLL0].SetChen(true)
	for !gclk.Gclk.Pchctrl[GCLK_OSCCTRL_FDPLL0].GetChen() {
	}

	oscctrl.Oscctrl.Dpll[0].Dpllctrla.SetEnable(false)
	oscctrl.Oscctrl.Dpll[0].Dpllctrlb.SetRefclk(oscctrl.DpllctrlbRefclkGclk)
	oscctrl.Oscctrl.Dpll[0].Dpllctrlb.SetLtime(oscctrl.DpllctrlbLtimeDefault)
	oscctrl.Oscctrl.Dpll[0].Dpllctrlb.SetFilter(oscctrl.DpllctrlbFilterFilter1)
	oscctrl.Oscctrl.Dpll[0].Dpllratio.SetLdrfrac(0)
	oscctrl.Oscctrl.Dpll[0].Dpllratio.SetLdr(119)
	oscctrl.Oscctrl.Dpll[0].Dpllctrla.SetOndemand(false)
	oscctrl.Oscctrl.Dpll[0].Dpllctrla.SetRunstdby(true)
	oscctrl.Oscctrl.Dpll[0].Dpllctrla.SetEnable(true)
	for !oscctrl.Oscctrl.Dpll[0].Dpllstatus.GetClkrdy() {
	}

	// Enable clock for the OSCCTRL
	mclk.Mclk.Apbamask.SetOscctrl(true)
	mclk.Mclk.Apbamask.SetGclk(true)
	mclk.Mclk.Apbamask.SetOsc32kctrl(true)
	mclk.Mclk.Apbbmask.SetPort(true)
	mclk.Mclk.Apbamask.SetEic(true)

	mclk.Mclk.Cpudiv.SetDiv(mclk.CpudivDivDiv1)
	for !mclk.Mclk.Intflag.GetCkrdy() {
	}

	// GCLK0 - 120MHz
	gclk.Gclk.Genctrl[GCLK0].SetRunstdby(true)
	for gclk.Gclk.Syncbusy.GetGenctrl()&gclk.SyncbusyGenctrlGclk0 != 0 {
	}

	gclk.Gclk.Genctrl[GCLK0].SetSrc(gclk.GenctrlSrcDpll0)
	for gclk.Gclk.Syncbusy.GetGenctrl()&gclk.SyncbusyGenctrlGclk0 != 0 {
	}

	gclk.Gclk.Genctrl[GCLK0].SetDiv(1)
	for gclk.Gclk.Syncbusy.GetGenctrl()&gclk.SyncbusyGenctrlGclk0 != 0 {
	}

	gclk.Gclk.Genctrl[GCLK0].SetDivsel(gclk.GenctrlDivselDiv1)
	for gclk.Gclk.Syncbusy.GetGenctrl()&gclk.SyncbusyGenctrlGclk0 != 0 {
	}

	gclk.Gclk.Genctrl[GCLK0].SetGenen(true)
	for gclk.Gclk.Syncbusy.GetGenctrl()&gclk.SyncbusyGenctrlGclk0 != 0 {
	}

	// GCLK1 - 60MHz
	gclk.Gclk.Genctrl[GCLK1].SetRunstdby(true)
	for gclk.Gclk.Syncbusy.GetGenctrl()&gclk.SyncbusyGenctrlGclk1 != 0 {
	}

	gclk.Gclk.Genctrl[GCLK1].SetSrc(gclk.GenctrlSrcDpll0)
	for gclk.Gclk.Syncbusy.GetGenctrl()&gclk.SyncbusyGenctrlGclk1 != 0 {
	}

	gclk.Gclk.Genctrl[GCLK1].SetDiv(2)
	for gclk.Gclk.Syncbusy.GetGenctrl()&gclk.SyncbusyGenctrlGclk1 != 0 {
	}

	gclk.Gclk.Genctrl[GCLK1].SetDivsel(gclk.GenctrlDivselDiv1)
	for gclk.Gclk.Syncbusy.GetGenctrl()&gclk.SyncbusyGenctrlGclk1 != 0 {
	}

	gclk.Gclk.Genctrl[GCLK1].SetGenen(true)
	for gclk.Gclk.Syncbusy.GetGenctrl()&gclk.SyncbusyGenctrlGclk1 != 0 {
	}

	/*
		// Choose the clock source for PORT
		gclk.Gclk.Pchctrl[11].SetChen(false)
		for gclk.Gclk.Pchctrl[11].GetChen() {
		}

		gclk.Gclk.Pchctrl[11].SetGen(chip.GCLK_PCHCTRL_REG_GEN_GCLK0)
		gclk.Gclk.Pchctrl[11].SetChen(true)
		for !gclk.Gclk.Pchctrl[11].GetChen() {
		}
	*/

	// Choose the clock source for EIC
	gclk.Gclk.Pchctrl[GCLK_EIC].SetChen(false)
	for gclk.Gclk.Pchctrl[GCLK_EIC].GetChen() {
	}

	gclk.Gclk.Pchctrl[GCLK_EIC].SetGen(gclk.PchctrlGenGclk0)
	gclk.Gclk.Pchctrl[GCLK_EIC].SetChen(true)
	for !gclk.Gclk.Pchctrl[GCLK_EIC].GetChen() {
	}

	// Choose the clock source for SERCOM0
	gclk.Gclk.Pchctrl[GCLK_SERCOM0_CORE].SetChen(false)
	for gclk.Gclk.Pchctrl[GCLK_SERCOM0_CORE].GetChen() {
	}

	gclk.Gclk.Pchctrl[GCLK_SERCOM0_CORE].SetGen(gclk.PchctrlGenGclk1)
	gclk.Gclk.Pchctrl[GCLK_SERCOM0_CORE].SetChen(true)
	for !gclk.Gclk.Pchctrl[GCLK_SERCOM0_CORE].GetChen() {
	}

	// Choose the clock source for SERCOM1
	gclk.Gclk.Pchctrl[GCLK_SERCOM1_CORE].SetChen(false)
	for gclk.Gclk.Pchctrl[GCLK_SERCOM1_CORE].GetChen() {
	}

	gclk.Gclk.Pchctrl[GCLK_SERCOM1_CORE].SetGen(gclk.PchctrlGenGclk1)
	gclk.Gclk.Pchctrl[GCLK_SERCOM1_CORE].SetChen(true)
	for !gclk.Gclk.Pchctrl[GCLK_SERCOM1_CORE].GetChen() {
	}

	// Choose the clock source for SERCOM2
	gclk.Gclk.Pchctrl[GCLK_SERCOM2_CORE].SetChen(false)
	for gclk.Gclk.Pchctrl[GCLK_SERCOM2_CORE].GetChen() {
	}

	gclk.Gclk.Pchctrl[GCLK_SERCOM2_CORE].SetGen(gclk.PchctrlGenGclk1)
	gclk.Gclk.Pchctrl[GCLK_SERCOM2_CORE].SetChen(true)
	for !gclk.Gclk.Pchctrl[GCLK_SERCOM2_CORE].GetChen() {
	}
	// Choose the clock source for SERCOM3
	gclk.Gclk.Pchctrl[GCLK_SERCOM3_CORE].SetChen(false)
	for gclk.Gclk.Pchctrl[GCLK_SERCOM3_CORE].GetChen() {
	}

	gclk.Gclk.Pchctrl[GCLK_SERCOM3_CORE].SetGen(gclk.PchctrlGenGclk1)
	gclk.Gclk.Pchctrl[GCLK_SERCOM3_CORE].SetChen(true)
	for !gclk.Gclk.Pchctrl[GCLK_SERCOM3_CORE].GetChen() {
	}

	// Choose the clock source for SERCOM4
	gclk.Gclk.Pchctrl[GCLK_SERCOM4_CORE].SetChen(false)
	for gclk.Gclk.Pchctrl[GCLK_SERCOM4_CORE].GetChen() {
	}

	gclk.Gclk.Pchctrl[GCLK_SERCOM4_CORE].SetGen(gclk.PchctrlGenGclk1)
	gclk.Gclk.Pchctrl[GCLK_SERCOM4_CORE].SetChen(true)
	for !gclk.Gclk.Pchctrl[GCLK_SERCOM4_CORE].GetChen() {
	}

	// Choose the clock source for SERCOM5
	gclk.Gclk.Pchctrl[GCLK_SERCOM5_CORE].SetChen(false)
	for gclk.Gclk.Pchctrl[GCLK_SERCOM5_CORE].GetChen() {
	}

	gclk.Gclk.Pchctrl[GCLK_SERCOM5_CORE].SetGen(gclk.PchctrlGenGclk1)
	gclk.Gclk.Pchctrl[GCLK_SERCOM5_CORE].SetChen(true)
	for !gclk.Gclk.Pchctrl[GCLK_SERCOM5_CORE].GetChen() {
	}

	// Choose the clock source for SERCOM6
	gclk.Gclk.Pchctrl[GCLK_SERCOM6_CORE].SetChen(false)
	for gclk.Gclk.Pchctrl[GCLK_SERCOM6_CORE].GetChen() {
	}

	gclk.Gclk.Pchctrl[GCLK_SERCOM6_CORE].SetGen(gclk.PchctrlGenGclk1)
	gclk.Gclk.Pchctrl[GCLK_SERCOM6_CORE].SetChen(true)
	for !gclk.Gclk.Pchctrl[GCLK_SERCOM6_CORE].GetChen() {
	}

	// Choose the clock source for SERCOM7
	gclk.Gclk.Pchctrl[GCLK_SERCOM7_CORE].SetChen(false)
	for gclk.Gclk.Pchctrl[GCLK_SERCOM7_CORE].GetChen() {
	}

	gclk.Gclk.Pchctrl[GCLK_SERCOM7_CORE].SetGen(gclk.PchctrlGenGclk1)
	gclk.Gclk.Pchctrl[GCLK_SERCOM7_CORE].SetChen(true)
	for !gclk.Gclk.Pchctrl[GCLK_SERCOM7_CORE].GetChen() {
	}
}

func InitFPU() {
	systemcontrol.Systemcontrol.Cpacr.SetCp10(systemcontrol.CpacrCp10Full)
	systemcontrol.Systemcontrol.Cpacr.SetCp11(systemcontrol.CpacrCp11Full)
}
