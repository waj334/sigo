//go:build atsamd21

package atsamd21

import (
	"runtime"
	"runtime/arm/cortexm"
	"runtime/arm/cortexm/sam/atsamd21/support/gclk"
	"runtime/arm/cortexm/sam/atsamd21/support/nvmctrl"
	"runtime/arm/cortexm/sam/atsamd21/support/pm"
	"runtime/arm/cortexm/sam/atsamd21/support/sysctrl"
	"unsafe"
)

var (
	SercomRefFrequency uint32 = 48_000_000
	Gclk0Frequency     uint32 = 48_000_000
)

func init() {
	cortexm.SysTickFrequency = Gclk0Frequency
	cortexm.IrqPriorityMask = 0b1111
}

func DefaultClocks() {
	state := runtime.DisableInterrupts()

	// Set flash wait states for 48 MHz
	nvmctrl.Nvmctrl.Ctrlb.SetRws(nvmctrl.CtrlbRwsHalf)

	initSYSCTRL()
	initGCLK3()
	initGCLK1()
	initDFLL()
	initGCLK0()
	initPM()
	initSERCOMCLK()

	runtime.EnableInterrupts(state)
}

func initSYSCTRL() {
	// Set up OSC8M
	var OSC8M sysctrl.RegisterOsc8mType
	OSC8M.SetPresc(sysctrl.Osc8mPresc0)
	OSC8M.SetOndemand(false)
	OSC8M.SetEnable(true)
	sysctrl.Sysctrl.Osc8m = OSC8M // Perform single write
	for !sysctrl.Sysctrl.Pclksr.GetOsc8mrdy() {
		// Wait for OSC8M to stabilize
	}

	// Set up XOSC32K
	sysctrl.Sysctrl.Xosc32k.SetEnable(false) // Enable XOSC32K separate
	var XOSC32K sysctrl.RegisterXosc32kType
	XOSC32K.SetStartup(sysctrl.Xosc32kStartupCycle2048) // 3 cycle start-up time
	XOSC32K.SetOndemand(false)                          // XOSC32K is always enabled
	XOSC32K.SetRunstdby(false)                          // XOSC32K will be disabled during sleep
	XOSC32K.SetAampen(false)                            // Disable automatic amplitude control
	XOSC32K.SetEn32k(true)                              // 32 KHz output is enabled
	XOSC32K.SetXtalen(true)                             // Enable external crystal
	sysctrl.Sysctrl.Xosc32k = XOSC32K                   // Perform single write
	sysctrl.Sysctrl.Xosc32k.SetEnable(true)             // Enable XOSC32K separate
	for !sysctrl.Sysctrl.Pclksr.GetXosc32krdy() {
		// Wait for XOSC32K to stabilize
	}
}

func initGCLK0() {
	// Set up GCLK0 with DFLL48 as the clock source
	var GENDIV0 gclk.RegisterGendivType
	GENDIV0.SetDiv(1)          // Divide by 1
	GENDIV0.SetId(0)           // Write configuration to GCLK3
	gclk.Gclk.Gendiv = GENDIV0 // Perform single write

	// Set up GCLK0
	var GENCTRL0 gclk.RegisterGenctrlType
	GENCTRL0.SetIdc(true)                   // The generic clock generator duty cycle is 50/50.
	GENCTRL0.SetSrc(gclk.GenctrlSrcDfll48m) // Use DFLL48M as the clock source.
	GENCTRL0.SetGenen(true)                 // The generic clock generator is enabled.
	GENCTRL0.SetId(0)                       // Write configuration to GCLK0
	gclk.Gclk.Genctrl = GENCTRL0            // Perform single write
	for gclk.Gclk.Status.GetSyncbusy() {
		// Wait for write to complete
	}
}

func initGCLK1() {
	var GENDIV1 gclk.RegisterGendivType
	GENDIV1.SetId(1)  // Write configuration to GCLK1
	GENDIV1.SetDiv(1) // Divide by 1
	gclk.Gclk.Gendiv = GENDIV1

	// Use XOSC32K as the source clock for GCLK1
	var GENCTRL1 gclk.RegisterGenctrlType
	GENCTRL1.SetId(1)                       // Write configuration to GCLK1
	GENCTRL1.SetGenen(true)                 // The generic clock generator is enabled.
	GENCTRL1.SetIdc(true)                   // The generic clock generator duty cycle is 50/50.
	GENCTRL1.SetSrc(gclk.GenctrlSrcXosc32k) // Use XOSC32K as the clock source.
	gclk.Gclk.Genctrl = GENCTRL1            // Perform single write
	for gclk.Gclk.Status.GetSyncbusy() {
		// Wait for write to complete
	}
}

func initGCLK3() {
	var GENDIV3 gclk.RegisterGendivType
	GENDIV3.SetDiv(1)          // Divide by 1
	GENDIV3.SetId(3)           // Write configuration to GCLK3
	gclk.Gclk.Gendiv = GENDIV3 // Perform single write

	// Set up GCLK3 with OSC8M as the clock source
	var GENCTRL3 gclk.RegisterGenctrlType
	GENCTRL3.SetRunstdby(false)                // Disable during standby
	GENCTRL3.SetDivsel(gclk.GenctrlDivselDiv1) // The generic clock generator equals the clock source divided by GENDIV.DIV.
	GENCTRL3.SetOe(false)                      // Disable generator output
	GENCTRL3.SetOov(false)                     // The GCLK_IO will be zero when the generic clock generator is turned off or when the OE bit is zero.
	GENCTRL3.SetIdc(true)                      // The generic clock generator duty cycle is 50/50.
	GENCTRL3.SetSrc(gclk.GenctrlSrcOsc8m)      // Use OSC8M as the clock source.
	GENCTRL3.SetGenen(true)                    // The generic clock generator is enabled.
	GENCTRL3.SetId(3)                          // Write configuration to GCLK3
	gclk.Gclk.Genctrl = GENCTRL3               // Perform single write
	for gclk.Gclk.Status.GetSyncbusy() {
		// Wait for write to complete
	}
}

func initDFLL() {
	// Errata 1.2.1: Write a '0' to the DFLL ONDEMAND bit in the DFLLCTRL register before configuring the DFLL module.
	sysctrl.Sysctrl.Dfllctrl.SetOndemand(false)
	for !sysctrl.Sysctrl.Pclksr.GetDfllrdy() {
		// Wait for DFLL to synchronize
	}

	// Load the DFLL48M coarse factory calibration value
	DFLL48_CALIB := (*uint32)(unsafe.Pointer(uintptr(0x806024))) // Load the second 32-bit word
	val := uint8((*DFLL48_CALIB >> 26) & 0x3F)
	if val == 0x3F {
		val = 0x1F
	}

	var DFLLVAL sysctrl.RegisterDfllvalType
	DFLLVAL.SetCoarse(val)
	DFLLVAL.SetFine(512)
	sysctrl.Sysctrl.Dfllval = DFLLVAL // Perform single write
	for !sysctrl.Sysctrl.Pclksr.GetDfllrdy() {
		// Wait for DFLL to synchronize
	}

	// Use GCLK1 as the source for Generic Clock Multiplexer 0 (DFLL48M reference)
	var CLKCTRL1 gclk.RegisterClkctrlType
	CLKCTRL1.SetId(gclk.ClkctrlIdDfll48)
	CLKCTRL1.SetGen(gclk.ClkctrlGenGclk1)
	CLKCTRL1.SetClken(true)
	gclk.Gclk.Clkctrl = CLKCTRL1 // Perform single write

	// Set up the multiplier for DFLL
	var DFLLMUL sysctrl.RegisterDfllmulType
	DFLLMUL.SetCstep(1)
	DFLLMUL.SetFstep(1)
	DFLLMUL.SetMul(1464)
	sysctrl.Sysctrl.Dfllmul = DFLLMUL
	for !sysctrl.Sysctrl.Pclksr.GetDfllrdy() {
		// Wait for DFLL to synchronize
	}

	// Enable DFLL48M
	var DFLLCTRL sysctrl.RegisterDfllctrlType
	DFLLCTRL.SetMode(true)
	// DFLLCTRL.SetWaitlock(true)
	DFLLCTRL.SetEnable(true)
	sysctrl.Sysctrl.Dfllctrl = DFLLCTRL
	for !sysctrl.Sysctrl.Pclksr.GetDflllckc() || !sysctrl.Sysctrl.Pclksr.GetDflllckc() {
		// Wait for frequency to lock
	}
}

func initPM() {
	pm.Pm.Cpusel.SetCpudiv(pm.CpuselCpudivDiv1)
	pm.Pm.Apbasel.SetApbadiv(pm.ApbaselApbadivDiv1)
	pm.Pm.Apbbsel.SetApbbdiv(pm.ApbbselApbbdivDiv1)
	pm.Pm.Apbcsel.SetApbcdiv(pm.ApbcselApbcdivDiv1)
}

func initSERCOMCLK() {
	var CLKCTRL gclk.RegisterClkctrlType
	// Set the source clock of each SERCOM to GCLK0
	CLKCTRL.SetGen(gclk.ClkctrlGenGclk0)
	CLKCTRL.SetClken(true)

	// SERCOM0
	CLKCTRL.SetId(gclk.ClkctrlIdSercom0Core)
	gclk.Gclk.Clkctrl = CLKCTRL

	// SERCOM1
	CLKCTRL.SetId(gclk.ClkctrlIdSercom1Core)
	gclk.Gclk.Clkctrl = CLKCTRL

	// SERCOM2
	CLKCTRL.SetId(gclk.ClkctrlIdSercom2Core)
	gclk.Gclk.Clkctrl = CLKCTRL

	// SERCOM3
	CLKCTRL.SetId(gclk.ClkctrlIdSercom3Core)
	gclk.Gclk.Clkctrl = CLKCTRL

	// SERCOM4
	CLKCTRL.SetId(gclk.ClkctrlIdSercom4Core)
	gclk.Gclk.Clkctrl = CLKCTRL

	// SERCOM5
	CLKCTRL.SetId(gclk.ClkctrlIdSercom5Core)
	gclk.Gclk.Clkctrl = CLKCTRL
}
