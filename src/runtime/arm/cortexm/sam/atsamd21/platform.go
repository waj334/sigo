package atsamd21

import (
	"runtime"
	"runtime/arm/cortexm"
	"runtime/arm/cortexm/sam/chip"
	"unsafe"
)

var (
	SERCOM_REF_FREQUENCY uint32 = 48_000_000
	GCLK0_FREQUENCY      uint32 = 48_000_000
)

func init() {
	cortexm.SYSTICK_FREQUENCY = GCLK0_FREQUENCY
}

func DefaultClocks() {
	state := runtime.DisableInterrupts()

	// Set flash wait states for 48 MHz
	chip.NVMCTRL.CTRLB.SetRWS(chip.NVMCTRL_CTRLB_REG_RWS_HALF)

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
	var OSC8M chip.SYSCTRL_OSC8M_REG
	OSC8M.SetPRESC(chip.SYSCTRL_OSC8M_REG_PRESC_0)
	OSC8M.SetONDEMAND(false)
	OSC8M.SetENABLE(true)
	chip.SYSCTRL.OSC8M = OSC8M // Peform single write
	for !chip.SYSCTRL.PCLKSR.GetOSC8MRDY() {
		// Wait for OSC8M to stabilize
	}

	// Set up XOSC32K
	chip.SYSCTRL.XOSC32K.SetENABLE(false) // Enable XOSC32K separate
	var XOSC32K chip.SYSCTRL_XOSC32K_REG
	XOSC32K.SetSTARTUP(chip.SYSCTRL_XOSC32K_REG_STARTUP_CYCLE2048) // 3 cycle start-up time
	XOSC32K.SetONDEMAND(false)                                     // XOSC32K is always enabled
	XOSC32K.SetRUNSTDBY(false)                                     // XOSC32K will be disabled during sleep
	XOSC32K.SetAAMPEN(false)                                       // Disable automatic amplitude control
	XOSC32K.SetEN32K(true)                                         // 32 KHz output is enabled
	XOSC32K.SetXTALEN(true)                                        // Enable external crystal
	chip.SYSCTRL.XOSC32K = XOSC32K                                 // Perform single write
	chip.SYSCTRL.XOSC32K.SetENABLE(true)                           // Enable XOSC32K separate
	for !chip.SYSCTRL.PCLKSR.GetXOSC32KRDY() {
		// Wait for XOSC32K to stabilize
	}
}

func initGCLK3() {
	var GENDIV3 chip.GCLK_GENDIV_REG
	GENDIV3.SetDIV(1)          // Divide by 1
	GENDIV3.SetID(3)           // Write configuration to GCLK3
	chip.GCLK.GENDIV = GENDIV3 // Perform single write

	// Set up GCLK3 with OSC8M as the clock source
	var GENCTRL3 chip.GCLK_GENCTRL_REG
	GENCTRL3.SetRUNSTDBY(false)                           // Disable during standby
	GENCTRL3.SetDIVSEL(chip.GCLK_GENCTRL_REG_DIVSEL_DIV1) // The generic clock generator equals the clock source divided by GENDIV.DIV.
	GENCTRL3.SetOE(false)                                 // Disable generator output
	GENCTRL3.SetOOV(false)                                // The GCLK_IO will be zero when the generic clock generator is turned off or when the OE bit is zero.
	GENCTRL3.SetIDC(true)                                 // The generic clock generator duty cycle is 50/50.
	GENCTRL3.SetSRC(chip.GCLK_GENCTRL_REG_SRC_OSC8M)      // Use OSC8M as the clock source.
	GENCTRL3.SetGENEN(true)                               // The generic clock generator is enabled.
	GENCTRL3.SetID(3)                                     // Write configuration to GCLK3
	chip.GCLK.GENCTRL = GENCTRL3                          // Perform single write
	for chip.GCLK.STATUS.GetSYNCBUSY() {
		// Wait for write to complete
	}
}

func initGCLK1() {
	var GENDIV1 chip.GCLK_GENDIV_REG
	GENDIV1.SetID(1)  // Write configuration to GCLK1
	GENDIV1.SetDIV(1) // Divide by 1
	chip.GCLK.GENDIV = GENDIV1

	// Use XOSC32K as the source clock for GCLK1
	var GENCTRL1 chip.GCLK_GENCTRL_REG
	GENCTRL1.SetID(1)                                  // Write configuration to GCLK1
	GENCTRL1.SetGENEN(true)                            // The generic clock generator is enabled.
	GENCTRL1.SetIDC(true)                              // The generic clock generator duty cycle is 50/50.
	GENCTRL1.SetSRC(chip.GCLK_GENCTRL_REG_SRC_XOSC32K) // Use XOSC32K as the clock source.
	chip.GCLK.GENCTRL = GENCTRL1                       // Perform single write
	for chip.GCLK.STATUS.GetSYNCBUSY() {
		// Wait for write to complete
	}
}

func initDFLL() {
	// Errata 1.2.1: Write a '0' to the DFLL ONDEMAND bit in the DFLLCTRL register before configuring the DFLL module.
	chip.SYSCTRL.DFLLCTRL.SetONDEMAND(false)
	for !chip.SYSCTRL.PCLKSR.GetDFLLRDY() {
		// Wait for DFLL to synchronize
	}

	// Load the DFLL48M coarse factory calibration value
	DFLL48_CALIB := (*uint32)(unsafe.Pointer(uintptr(0x806024))) // Load the second 32-bit word
	val := uint8((*DFLL48_CALIB >> 26) & 0x3F)
	if val == 0x3F {
		val = 0x1F
	}

	var DFLLVAL chip.SYSCTRL_DFLLVAL_REG
	DFLLVAL.SetCOARSE(val)
	DFLLVAL.SetFINE(512)
	chip.SYSCTRL.DFLLVAL = DFLLVAL // Perform single write
	for !chip.SYSCTRL.PCLKSR.GetDFLLRDY() {
		// Wait for DFLL to synchronize
	}

	// Use GCLK1 as the source for Generic Clock Multiplexer 0 (DFLL48M reference)
	var CLKCTRL1 chip.GCLK_CLKCTRL_REG
	CLKCTRL1.SetID(chip.GCLK_CLKCTRL_REG_ID_DFLL48)
	CLKCTRL1.SetGEN(chip.GCLK_CLKCTRL_REG_GEN_GCLK1)
	CLKCTRL1.SetCLKEN(true)
	chip.GCLK.CLKCTRL = CLKCTRL1 // Perform single write

	// Set up the multiplier for DFLL
	var DFLLMUL chip.SYSCTRL_DFLLMUL_REG
	DFLLMUL.SetCSTEP(1)
	DFLLMUL.SetFSTEP(1)
	DFLLMUL.SetMUL(1464)
	chip.SYSCTRL.DFLLMUL = DFLLMUL
	for !chip.SYSCTRL.PCLKSR.GetDFLLRDY() {
		// Wait for DFLL to synchronize
	}

	// Enable DFLL48M
	var DFLLCTRL chip.SYSCTRL_DFLLCTRL_REG
	DFLLCTRL.SetMODE(true)
	//DFLLCTRL.SetWAITLOCK(true)
	DFLLCTRL.SetENABLE(true)
	chip.SYSCTRL.DFLLCTRL = DFLLCTRL
	for !chip.SYSCTRL.PCLKSR.GetDFLLLCKC() || !chip.SYSCTRL.PCLKSR.GetDFLLLCKC() {
		// Wait for frequency to lock
	}
}

func initGCLK0() {
	// Set up GCLK0 with DFLL48 as the clock source
	var GENDIV0 chip.GCLK_GENDIV_REG
	GENDIV0.SetDIV(1)          // Divide by 1
	GENDIV0.SetID(0)           // Write configuration to GCLK3
	chip.GCLK.GENDIV = GENDIV0 // Perform single write

	// Set up GCLK0
	var GENCTRL0 chip.GCLK_GENCTRL_REG
	GENCTRL0.SetIDC(true)                              // The generic clock generator duty cycle is 50/50.
	GENCTRL0.SetSRC(chip.GCLK_GENCTRL_REG_SRC_DFLL48M) // Use DFLL48M as the clock source.
	GENCTRL0.SetGENEN(true)                            // The generic clock generator is enabled.
	GENCTRL0.SetID(0)                                  // Write configuration to GCLK0
	chip.GCLK.GENCTRL = GENCTRL0                       // Perform single write
	for chip.GCLK.STATUS.GetSYNCBUSY() {
		// Wait for write to complete
	}
}

func initPM() {
	chip.PM.CPUSEL.SetCPUDIV(chip.PM_CPUSEL_REG_CPUDIV_DIV1)
	chip.PM.APBASEL.SetAPBADIV(chip.PM_APBASEL_REG_APBADIV_DIV1)
	chip.PM.APBBSEL.SetAPBBDIV(chip.PM_APBBSEL_REG_APBBDIV_DIV1)
	chip.PM.APBCSEL.SetAPBCDIV(chip.PM_APBCSEL_REG_APBCDIV_DIV1)
}

func initSERCOMCLK() {
	var CLKCTRL chip.GCLK_CLKCTRL_REG
	// Set the source clock of each SERCOM to GCLK0
	CLKCTRL.SetGEN(chip.GCLK_CLKCTRL_REG_GEN_GCLK0)
	CLKCTRL.SetCLKEN(true)

	// SERCOM0
	CLKCTRL.SetID(chip.GCLK_CLKCTRL_REG_ID_SERCOM0_CORE)
	chip.GCLK.CLKCTRL = CLKCTRL

	// SERCOM1
	CLKCTRL.SetID(chip.GCLK_CLKCTRL_REG_ID_SERCOM1_CORE)
	chip.GCLK.CLKCTRL = CLKCTRL

	// SERCOM2
	CLKCTRL.SetID(chip.GCLK_CLKCTRL_REG_ID_SERCOM2_CORE)
	chip.GCLK.CLKCTRL = CLKCTRL

	// SERCOM3
	CLKCTRL.SetID(chip.GCLK_CLKCTRL_REG_ID_SERCOM3_CORE)
	chip.GCLK.CLKCTRL = CLKCTRL

	// SERCOM4
	CLKCTRL.SetID(chip.GCLK_CLKCTRL_REG_ID_SERCOM4_CORE)
	chip.GCLK.CLKCTRL = CLKCTRL

	// SERCOM5
	CLKCTRL.SetID(chip.GCLK_CLKCTRL_REG_ID_SERCOM5_CORE)
	chip.GCLK.CLKCTRL = CLKCTRL
}
