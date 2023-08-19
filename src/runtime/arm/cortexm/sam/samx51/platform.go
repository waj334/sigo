package samx51

import (
	"runtime/arm/cortexm"
	"runtime/arm/cortexm/sam/chip"
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
	chip.OSC32KCTRL.XOSC32K.SetCGM(chip.OSC32KCTRL_XOSC32K_REG_CGM_XT)
	chip.OSC32KCTRL.XOSC32K.SetXTALEN(true)
	chip.OSC32KCTRL.XOSC32K.SetEN32K(true)
	chip.OSC32KCTRL.XOSC32K.SetONDEMAND(false)
	chip.OSC32KCTRL.XOSC32K.SetRUNSTDBY(true)
	chip.OSC32KCTRL.XOSC32K.SetSTARTUP(chip.OSC32KCTRL_XOSC32K_REG_STARTUP_CYCLE2048)
	chip.OSC32KCTRL.XOSC32K.SetENABLE(true)

	// Wait for XOSC32K to be stable
	for !chip.OSC32KCTRL.INTFLAG.GetXOSC32KRDY() {
	}

	// Set RTC clock
	chip.OSC32KCTRL.RTCCTRL.SetRTCSEL(chip.OSC32KCTRL_RTCCTRL_REG_RTCSEL_XOSC32K)

	// Enable DFLL - 48MHz
	chip.OSCCTRL.DFLLCTRLA.SetONDEMAND(false)
	chip.OSCCTRL.DFLLCTRLA.SetENABLE(true)
	for chip.OSCCTRL.DFLLSYNC.GetENABLE() {
	}

	// Set up GCLK2 - 1MHz
	chip.GCLK.GENCTRL[2].SetSRC(chip.GCLK_GENCTRL_REG_SRC_DFLL)
	chip.GCLK.GENCTRL[2].SetDIV(48)
	chip.GCLK.GENCTRL[2].SetGENEN(true)
	for chip.GCLK.SYNCBUSY.GetGENCTRL()&chip.GCLK_SYNCBUSY_REG_GENCTRL_GCLK2 != 0 {
	}

	// Configure the DPLL - 120MHz
	chip.GCLK.PCHCTRL[1].SetCHEN(false)
	for chip.GCLK.PCHCTRL[1].GetCHEN() {
	}

	chip.GCLK.PCHCTRL[1].SetGEN(chip.GCLK_PCHCTRL_REG_GEN_GCLK2)
	chip.GCLK.PCHCTRL[1].SetCHEN(true)
	for !chip.GCLK.PCHCTRL[1].GetCHEN() {
	}

	chip.OSCCTRL.DPLL[0].DPLLCTRLB.SetREFCLK(chip.OSCCTRL_DPLLCTRLB_REG_REFCLK_GCLK)
	chip.OSCCTRL.DPLL[0].DPLLCTRLB.SetLTIME(chip.OSCCTRL_DPLLCTRLB_REG_LTIME_DEFAULT)
	chip.OSCCTRL.DPLL[0].DPLLCTRLB.SetFILTER(chip.OSCCTRL_DPLLCTRLB_REG_FILTER_FILTER1)

	chip.OSCCTRL.DPLL[0].DPLLRATIO.SetLDRFRAC(0)
	chip.OSCCTRL.DPLL[0].DPLLRATIO.SetLDR(119)
	for chip.OSCCTRL.DPLL[0].DPLLSYNCBUSY.GetDPLLRATIO() {
	}

	chip.OSCCTRL.DPLL[0].DPLLCTRLA.SetONDEMAND(false)
	chip.OSCCTRL.DPLL[0].DPLLCTRLA.SetRUNSTDBY(true)
	chip.OSCCTRL.DPLL[0].DPLLCTRLA.SetENABLE(true)

	// Wait for DPLL to be ready
	for !chip.OSCCTRL.DPLL[0].DPLLSTATUS.GetCLKRDY() {
	}

	// Enable clock for the OSCCTRL
	chip.MCLK.APBAMASK.SetOSCCTRL(true)
	chip.MCLK.APBAMASK.SetGCLK(true)
	chip.MCLK.APBAMASK.SetOSC32KCTRL(true)
	chip.MCLK.APBBMASK.SetPORT(true)
	chip.MCLK.APBAMASK.SetEIC(true)

	chip.MCLK.CPUDIV.SetDIV(chip.MCLK_CPUDIV_REG_DIV_DIV1)
	for !chip.MCLK.INTFLAG.GetCKRDY() {
	}

	// GCLK0 - 120MHz
	chip.GCLK.GENCTRL[0].SetRUNSTDBY(true)
	chip.GCLK.GENCTRL[0].SetSRC(chip.GCLK_GENCTRL_REG_SRC_DPLL0)
	chip.GCLK.GENCTRL[0].SetDIV(1)
	chip.GCLK.GENCTRL[0].SetDIVSEL(0)
	chip.GCLK.GENCTRL[0].SetGENEN(true)
	for chip.GCLK.SYNCBUSY.GetGENCTRL()&chip.GCLK_SYNCBUSY_REG_GENCTRL_GCLK0 != 0 {
	}

	// GCLK1 - 60MHz
	chip.GCLK.GENCTRL[1].SetRUNSTDBY(true)
	chip.GCLK.GENCTRL[1].SetSRC(chip.GCLK_GENCTRL_REG_SRC_DPLL0)
	chip.GCLK.GENCTRL[1].SetDIV(2)
	chip.GCLK.GENCTRL[1].SetDIVSEL(0)
	chip.GCLK.GENCTRL[1].SetGENEN(true)
	for chip.GCLK.SYNCBUSY.GetGENCTRL()&chip.GCLK_SYNCBUSY_REG_GENCTRL_GCLK1 != 0 {
	}

	// Choose the clock source for PORT
	chip.GCLK.PCHCTRL[11].SetCHEN(false)
	for chip.GCLK.PCHCTRL[11].GetCHEN() {
	}

	chip.GCLK.PCHCTRL[11].SetCHEN(true)
	chip.GCLK.PCHCTRL[11].SetGEN(chip.GCLK_PCHCTRL_REG_GEN_GCLK0)
	for !chip.GCLK.PCHCTRL[11].GetCHEN() {
	}

	// Choose the clock source for EIC
	chip.GCLK.PCHCTRL[4].SetCHEN(false)
	for chip.GCLK.PCHCTRL[4].GetCHEN() {
	}

	chip.GCLK.PCHCTRL[4].SetCHEN(true)
	chip.GCLK.PCHCTRL[4].SetGEN(chip.GCLK_PCHCTRL_REG_GEN_GCLK0)
	for !chip.GCLK.PCHCTRL[4].GetCHEN() {
	}

	// Choose the clock source for SERCOM0
	chip.GCLK.PCHCTRL[7].SetCHEN(false)
	for chip.GCLK.PCHCTRL[7].GetCHEN() {
	}

	chip.GCLK.PCHCTRL[7].SetCHEN(true)
	chip.GCLK.PCHCTRL[7].SetGEN(chip.GCLK_PCHCTRL_REG_GEN_GCLK1)
	for !chip.GCLK.PCHCTRL[7].GetCHEN() {
	}

	// Choose the clock source for SERCOM1
	chip.GCLK.PCHCTRL[8].SetCHEN(false)
	for chip.GCLK.PCHCTRL[8].GetCHEN() {
	}

	chip.GCLK.PCHCTRL[8].SetCHEN(true)
	chip.GCLK.PCHCTRL[8].SetGEN(chip.GCLK_PCHCTRL_REG_GEN_GCLK1)
	for !chip.GCLK.PCHCTRL[8].GetCHEN() {
	}

	// Choose the clock source for SERCOM2
	chip.GCLK.PCHCTRL[23].SetCHEN(false)
	for chip.GCLK.PCHCTRL[23].GetCHEN() {
	}

	chip.GCLK.PCHCTRL[23].SetCHEN(true)
	chip.GCLK.PCHCTRL[23].SetGEN(chip.GCLK_PCHCTRL_REG_GEN_GCLK1)
	for !chip.GCLK.PCHCTRL[23].GetCHEN() {
	}
	// Choose the clock source for SERCOM3
	chip.GCLK.PCHCTRL[24].SetCHEN(false)
	for chip.GCLK.PCHCTRL[24].GetCHEN() {
	}

	chip.GCLK.PCHCTRL[24].SetCHEN(true)
	chip.GCLK.PCHCTRL[24].SetGEN(chip.GCLK_PCHCTRL_REG_GEN_GCLK1)
	for !chip.GCLK.PCHCTRL[24].GetCHEN() {
	}

	// Choose the clock source for SERCOM4
	chip.GCLK.PCHCTRL[34].SetCHEN(false)
	for chip.GCLK.PCHCTRL[34].GetCHEN() {
	}

	chip.GCLK.PCHCTRL[34].SetCHEN(true)
	chip.GCLK.PCHCTRL[34].SetGEN(chip.GCLK_PCHCTRL_REG_GEN_GCLK1)
	for !chip.GCLK.PCHCTRL[34].GetCHEN() {
	}

	// Choose the clock source for SERCOM5
	chip.GCLK.PCHCTRL[35].SetCHEN(false)
	for chip.GCLK.PCHCTRL[35].GetCHEN() {
	}

	chip.GCLK.PCHCTRL[35].SetCHEN(true)
	chip.GCLK.PCHCTRL[35].SetGEN(chip.GCLK_PCHCTRL_REG_GEN_GCLK1)
	for !chip.GCLK.PCHCTRL[35].GetCHEN() {
	}

	// Choose the clock source for SERCOM6
	chip.GCLK.PCHCTRL[36].SetCHEN(false)
	for chip.GCLK.PCHCTRL[36].GetCHEN() {
	}

	chip.GCLK.PCHCTRL[36].SetCHEN(true)
	chip.GCLK.PCHCTRL[36].SetGEN(chip.GCLK_PCHCTRL_REG_GEN_GCLK1)
	for !chip.GCLK.PCHCTRL[36].GetCHEN() {
	}

	// Choose the clock source for SERCOM7
	chip.GCLK.PCHCTRL[37].SetCHEN(false)
	for chip.GCLK.PCHCTRL[37].GetCHEN() {
	}

	chip.GCLK.PCHCTRL[37].SetCHEN(true)
	chip.GCLK.PCHCTRL[37].SetGEN(chip.GCLK_PCHCTRL_REG_GEN_GCLK1)
	for !chip.GCLK.PCHCTRL[37].GetCHEN() {
	}
}

func InitFPU() {
	chip.SystemControl.CPACR.SetCP10(chip.SystemControl_CPACR_REG_CP10_FULL)
	chip.SystemControl.CPACR.SetCP11(chip.SystemControl_CPACR_REG_CP11_FULL)
}
