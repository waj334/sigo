//sigo:architecture arm
//sigo:cpu cortex-m4
//sigo:triple armv7m-none-eabi

package main

import (
	"time"

	_ "omibyte.io/sigo/src/runtime/arm/cortexm/sam/atsame51g19a"
	mcu "omibyte.io/sigo/src/runtime/arm/cortexm/sam/atsamx51"
)

func init() {
	// Configure the XOSC32K oscillator
	mcu.OSC32KCTRL.XOSC32K.SetCGM(mcu.OSC32KCTRLXOSC32KCGMSelectXT)
	mcu.OSC32KCTRL.XOSC32K.SetXTALEN(true)
	mcu.OSC32KCTRL.XOSC32K.SetEN32K(true)
	mcu.OSC32KCTRL.XOSC32K.SetONDEMAND(false)
	mcu.OSC32KCTRL.XOSC32K.SetRUNSTDBY(true)
	mcu.OSC32KCTRL.XOSC32K.SetSTARTUP(mcu.OSC32KCTRLXOSC32KSTARTUPSelectCYCLE2048)
	mcu.OSC32KCTRL.XOSC32K.SetENABLE(true)

	// Wait for XOSC32K to be stable
	for !mcu.OSC32KCTRL.INTFLAG.GetXOSC32KRDY() {
	}

	// Set RTC clock
	mcu.OSC32KCTRL.RTCCTRL.SetRTCSEL(mcu.OSC32KCTRLRTCCTRLRTCSELSelectXOSC32K)

	// Enable DFLL
	mcu.OSCCTRL.DFLLCTRLA.SetONDEMAND(false)
	mcu.OSCCTRL.DFLLCTRLA.SetENABLE(true)
	for mcu.OSCCTRL.DFLLSYNC.GetENABLE() {
	}

	// Set up GCLK2
	mcu.GCLK.GENCTRL[2].SetSRC(mcu.GCLKGENCTRLSRCSelectDFLL)
	mcu.GCLK.GENCTRL[2].SetDIV(48)
	mcu.GCLK.GENCTRL[2].SetGENEN(true)
	for mcu.GCLK.SYNCBUSY.GetGENCTRL()&mcu.GCLKSYNCBUSYGENCTRLSelectGCLK2 != 0 {
	}

	// Configure the DPLL
	mcu.GCLK.PCHCTRL[1].SetCHEN(false)
	for mcu.GCLK.PCHCTRL[1].GetCHEN() {
	}

	mcu.GCLK.PCHCTRL[1].SetGEN(mcu.GCLKPCHCTRLGENSelectGCLK2)
	mcu.GCLK.PCHCTRL[1].SetCHEN(true)
	for !mcu.GCLK.PCHCTRL[1].GetCHEN() {
	}

	mcu.OSCCTRL.DPLL[0].DPLLCTRLB.SetREFCLK(mcu.OSCCTRLDPLLDPLLCTRLBREFCLKSelectGCLK)
	mcu.OSCCTRL.DPLL[0].DPLLCTRLB.SetLTIME(mcu.OSCCTRLDPLLDPLLCTRLBLTIMESelectDEFAULT)
	mcu.OSCCTRL.DPLL[0].DPLLCTRLB.SetFILTER(mcu.OSCCTRLDPLLDPLLCTRLBFILTERSelectFILTER1)

	mcu.OSCCTRL.DPLL[0].DPLLRATIO.SetLDRFRAC(0)
	mcu.OSCCTRL.DPLL[0].DPLLRATIO.SetLDR(119)
	for mcu.OSCCTRL.DPLL[0].DPLLSYNCBUSY.GetDPLLRATIO() {
	}

	mcu.OSCCTRL.DPLL[0].DPLLCTRLA.SetONDEMAND(false)
	mcu.OSCCTRL.DPLL[0].DPLLCTRLA.SetRUNSTDBY(true)
	mcu.OSCCTRL.DPLL[0].DPLLCTRLA.SetENABLE(true)

	// Wait for DPLL to be ready
	for !mcu.OSCCTRL.DPLL[0].DPLLSTATUS.GetCLKRDY() {
	}

	// Enable clock for the OSCCTRL
	mcu.MCLK.APBAMASK.SetOSCCTRL(true)
	mcu.MCLK.APBAMASK.SetGCLK(true)
	mcu.MCLK.APBAMASK.SetOSC32KCTRL(true)
	mcu.MCLK.APBBMASK.SetPORT(true)
	mcu.MCLK.APBAMASK.SetEIC(true)

	mcu.MCLK.CPUDIV.SetDIV(mcu.MCLKCPUDIVDIVSelectDIV1)
	for !mcu.MCLK.INTFLAG.GetCKRDY() {
	}

	mcu.GCLK.GENCTRL[0].SetRUNSTDBY(true)
	mcu.GCLK.GENCTRL[0].SetSRC(mcu.GCLKGENCTRLSRCSelectDPLL0)
	mcu.GCLK.GENCTRL[0].SetDIV(1)
	mcu.GCLK.GENCTRL[0].SetDIVSEL(0)
	mcu.GCLK.GENCTRL[0].SetGENEN(true)
	for mcu.GCLK.SYNCBUSY.GetGENCTRL()&mcu.GCLKSYNCBUSYGENCTRLSelectGCLK0 != 0 {
	}

	mcu.GCLK.GENCTRL[1].SetRUNSTDBY(true)
	for mcu.GCLK.SYNCBUSY.GetGENCTRL()&mcu.GCLKSYNCBUSYGENCTRLSelectGCLK1 != 0 {
	}

	mcu.GCLK.GENCTRL[1].SetSRC(mcu.GCLKGENCTRLSRCSelectDPLL0)
	for mcu.GCLK.SYNCBUSY.GetGENCTRL()&mcu.GCLKSYNCBUSYGENCTRLSelectGCLK1 != 0 {
	}

	mcu.GCLK.GENCTRL[1].SetDIV(1)
	for mcu.GCLK.SYNCBUSY.GetGENCTRL()&mcu.GCLKSYNCBUSYGENCTRLSelectGCLK1 != 0 {
	}

	mcu.GCLK.GENCTRL[1].SetGENEN(true)
	for mcu.GCLK.SYNCBUSY.GetGENCTRL()&mcu.GCLKSYNCBUSYGENCTRLSelectGCLK1 != 0 {
	}

	// Choose the clock source for PORT
	mcu.GCLK.PCHCTRL[11].SetCHEN(false)
	for mcu.GCLK.PCHCTRL[11].GetCHEN() {
	}

	mcu.GCLK.PCHCTRL[11].SetCHEN(true)
	mcu.GCLK.PCHCTRL[11].SetGEN(mcu.GCLKPCHCTRLGENSelectGCLK0)
	for !mcu.GCLK.PCHCTRL[11].GetCHEN() {
	}

	mcu.GCLK.PCHCTRL[4].SetCHEN(false)
	for mcu.GCLK.PCHCTRL[4].GetCHEN() {
	}

	mcu.GCLK.PCHCTRL[4].SetCHEN(true)
	mcu.GCLK.PCHCTRL[4].SetGEN(mcu.GCLKPCHCTRLGENSelectGCLK0)
	for !mcu.GCLK.PCHCTRL[4].GetCHEN() {
	}
}

func main() {
	pin := mcu.PB11
	pin.SetDirection(1)
	pin.Set(true)

	button := mcu.PB22
	button.SetDirection(0)
	button.SetInterrupt(2, func(pin mcu.Pin) {
		mcu.PB11.Toggle()
	})

	go func() {
		for {
			time.Sleep(time.Second)
			pin.Toggle()
		}
	}()

	for {
	}
}
