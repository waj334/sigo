package uart

import (
	"runtime/arm/cortexm/sam/atsamx51"
	"runtime/arm/cortexm/sam/atsamx51/internal"
	"runtime/arm/cortexm/sam/chip"
	"sync"
)

const (
	UsartFrame           = chip.SERCOMUSART_INTCTRLAFORMSelectUSART_FRAME_NO_PARITY
	UsartFrameWithParity = chip.SERCOMUSART_INTCTRLAFORMSelectUSART_FRAME_WITH_PARITY
)

var (
	UART0 = &UART{sercom: 0}
	UART1 = &UART{sercom: 1}
	UART2 = &UART{sercom: 2}
	UART3 = &UART{sercom: 3}
	UART4 = &UART{sercom: 4}
	UART5 = &UART{sercom: 5}
	UART6 = &UART{sercom: 6}
	UART7 = &UART{sercom: 7}
	uart  = [8]*UART{
		UART0,
		UART1,
		UART2,
		UART3,
		UART4,
		UART5,
		UART6,
		UART7,
	}
)

type UART struct {
	sercom   int
	txBuffer internal.RingBuffer
	rxBuffer internal.RingBuffer
	mutex    sync.Mutex
}

type Config struct {
	TXD             atsamx51.Pin
	RXD             atsamx51.Pin
	XCK             atsamx51.Pin
	RTS             atsamx51.Pin
	CTS             atsamx51.Pin
	FrameFormat     chip.SERCOMUSART_INTCTRLAFORMSelect
	BaudHz          uint
	CharacterSize   uint
	NumStopBits     uint
	ReceiveEnabled  bool
	TransmitEnabled bool
	RXBufferSize    uintptr
	TXBufferSize    uintptr
}

func (u *UART) Configure(config Config) {
	var mode atsamx51.PMUXFunction
	var rxPad int

	// Determine the SERCOM number from the PAD value of TXD pin
	if config.TXD.GetPAD() == 0 {
		mode = atsamx51.PMUXFunctionC
		rxPad = config.RXD.GetPAD()
		// Check the other optional pins
		if config.TXD.GetSERCOM() != u.sercom ||
			config.XCK != 0 && config.XCK.GetPAD() != 1 ||
			config.RTS != 0 && config.RTS.GetPAD() != 2 ||
			config.CTS != 0 && config.CTS.GetPAD() != 3 ||
			// The receive pad must not conflict with any of the other pads
			rxPad == config.TXD.GetPAD() ||
			rxPad == config.XCK.GetPAD() ||
			rxPad == config.RTS.GetPAD() ||
			rxPad == config.CTS.GetPAD() {
			panic("invalid selection")
		}
	} else if config.TXD.GetAltPAD() == 0 {
		mode = atsamx51.PMUXFunctionD
		rxPad = config.RXD.GetAltPAD()
		// Check the other optional pins
		if config.TXD.GetAltSERCOM() != u.sercom ||
			config.XCK != 0 && config.XCK.GetAltPAD() != 1 ||
			config.RTS != 0 && config.RTS.GetAltPAD() != 2 ||
			config.CTS != 0 && config.CTS.GetAltPAD() != 3 ||
			// The receive pad must not conflict with any of the other pads
			rxPad == config.TXD.GetAltPAD() ||
			rxPad == config.XCK.GetAltPAD() ||
			rxPad == config.RTS.GetAltPAD() ||
			rxPad == config.CTS.GetAltPAD() {
			panic("invalid selection")
		}
	} else {
		panic("invalid pin selection")
	}

	// Set the pin configurations
	config.TXD.SetPMUX(mode, true)
	config.RXD.SetPMUX(mode, true)

	if config.XCK != 0 {
		config.XCK.SetPMUX(mode, true)
	}

	if config.RTS != 0 {
		config.RTS.SetPMUX(mode, true)
	}

	if config.CTS != 0 {
		config.CTS.SetPMUX(mode, true)
	}

	// Disable the SERCOM first
	chip.SERCOM[u.sercom].USART_INT.CTRLA.SetENABLE(false)
	for chip.SERCOM[u.sercom].USART_INT.SYNCBUSY.GetENABLE() {
	}

	// Enabled the SERCOM in MCLK
	switch u.sercom {
	case 0:
		chip.MCLK.APBAMASK.SetSERCOM0(true)
	case 1:
		chip.MCLK.APBAMASK.SetSERCOM1(true)
	case 2:
		chip.MCLK.APBBMASK.SetSERCOM2(true)
	case 3:
		chip.MCLK.APBBMASK.SetSERCOM3(true)
	case 4:
		chip.MCLK.APBDMASK.SetSERCOM4(true)
	case 5:
		chip.MCLK.APBDMASK.SetSERCOM5(true)
	case 6:
		chip.MCLK.APBDMASK.SetSERCOM6(true)
	case 7:
		chip.MCLK.APBDMASK.SetSERCOM7(true)
	}

	// Set the baud rate
	ratio := (uint64(atsamx51.SERCOM_REF_FREQUENCY) * uint64(1000)) / uint64(config.BaudHz*16)
	baud := ratio / 1000
	fp := ((ratio - (baud * 1000)) * 8) / 1000

	chip.SERCOM[u.sercom].USART_INT.CTRLA.SetSAMPR(chip.SERCOMUSART_INTCTRLASAMPRSelect16X_FRACTIONAL)
	chip.SERCOM[u.sercom].USART_INT.BAUD.SetBAUD(uint16(baud))
	chip.SERCOM[u.sercom].USART_INT.BAUD.SetFP(uint8(fp))

	chip.SERCOM[u.sercom].USART_INT.CTRLA.SetMODE(chip.SERCOMUSART_INTCTRLAMODESelectUSART_INT_CLK)
	chip.SERCOM[u.sercom].USART_INT.CTRLA.SetRXPO(chip.SERCOMUSART_INTCTRLARXPOSelect(rxPad))
	if config.XCK != 0 && config.RTS != 0 {
		chip.SERCOM[u.sercom].USART_INT.CTRLA.SetTXPO(chip.SERCOMUSART_INTCTRLATXPOSelectPAD3)
	} else if config.RTS != 0 && config.CTS != 0 {
		chip.SERCOM[u.sercom].USART_INT.CTRLA.SetTXPO(chip.SERCOMUSART_INTCTRLATXPOSelect(0x2))
	} else {
		chip.SERCOM[u.sercom].USART_INT.CTRLA.SetTXPO(chip.SERCOMUSART_INTCTRLATXPOSelectPAD0)
	}
	chip.SERCOM[u.sercom].USART_INT.CTRLA.SetDORD(chip.SERCOMUSART_INTCTRLADORDSelectLSB)
	chip.SERCOM[u.sercom].USART_INT.CTRLA.SetCMODE(chip.SERCOMUSART_INTCTRLACMODESelectASYNC)

	switch config.CharacterSize {
	case 5:
		chip.SERCOM[u.sercom].USART_INT.CTRLB.SetCHSIZE(chip.SERCOMUSART_INTCTRLBCHSIZESelect5_BIT)
	case 6:
		chip.SERCOM[u.sercom].USART_INT.CTRLB.SetCHSIZE(chip.SERCOMUSART_INTCTRLBCHSIZESelect6_BIT)
	case 7:
		chip.SERCOM[u.sercom].USART_INT.CTRLB.SetCHSIZE(chip.SERCOMUSART_INTCTRLBCHSIZESelect7_BIT)
	case 8:
		chip.SERCOM[u.sercom].USART_INT.CTRLB.SetCHSIZE(chip.SERCOMUSART_INTCTRLBCHSIZESelect8_BIT)
	case 9:
		chip.SERCOM[u.sercom].USART_INT.CTRLB.SetCHSIZE(chip.SERCOMUSART_INTCTRLBCHSIZESelect9_BIT)
	default:
		panic("invalid character size value")
	}
	for chip.SERCOM[u.sercom].USART_INT.SYNCBUSY.GetCTRLB() {
	}

	chip.SERCOM[u.sercom].USART_INT.CTRLA.SetFORM(config.FrameFormat)

	switch config.NumStopBits {
	case 1:
		chip.SERCOM[u.sercom].USART_INT.CTRLB.SetSBMODE(chip.SERCOMUSART_INTCTRLBSBMODESelect1_BIT)
	case 2:
		chip.SERCOM[u.sercom].USART_INT.CTRLB.SetSBMODE(chip.SERCOMUSART_INTCTRLBSBMODESelect2_BIT)
	default:
		panic("invalid stop bits value")
	}
	for chip.SERCOM[u.sercom].USART_INT.SYNCBUSY.GetCTRLB() {
	}

	if config.ReceiveEnabled {
		chip.SERCOM[u.sercom].USART_INT.CTRLB.SetRXEN(true)
		for chip.SERCOM[u.sercom].USART_INT.SYNCBUSY.GetCTRLB() {
		}
	}

	if config.TransmitEnabled {
		chip.SERCOM[u.sercom].USART_INT.CTRLB.SetTXEN(true)
		for chip.SERCOM[u.sercom].USART_INT.SYNCBUSY.GetCTRLB() {
		}
	}

	rx := internal.NewRingBuffer(config.RXBufferSize)
	tx := internal.NewRingBuffer(config.TXBufferSize)

	u.rxBuffer = rx
	u.txBuffer = tx
	atsamx51.SERCOMHandlers[u.sercom][0].Set(irqHandler)

	// Enable interrupts
	irqBase := 46 + u.sercom*4
	irq := atsamx51.Interrupt(irqBase)
	irq.EnableIRQ()
	chip.SERCOM[u.sercom].USART_INT.INTENSET.SetRXC(true)

	// Enable the peripheral
	chip.SERCOM[u.sercom].USART_INT.CTRLA.SetENABLE(true)
	for chip.SERCOM[u.sercom].USART_INT.SYNCBUSY.GetENABLE() {
	}
}

func irqHandler() {
	sercom := int(chip.SystemControl.ICSR.GetVECTACTIVE()-62) / 4
	switch {
	case chip.SERCOM[sercom].USART_INT.INTFLAG.GetRXC():
		rxcHandler(sercom)
	case chip.SERCOM[sercom].USART_INT.INTFLAG.GetDRE():
		dreHandler(sercom)
	}
}

func rxcHandler(sercom int) {
	b := byte(chip.SERCOM[sercom].USART_INT.DATA.GetDATA())
	uart[sercom].rxBuffer.WriteByte(b)
}

func dreHandler(sercom int) {
	for uart[sercom].txBuffer.Len() > 0 {
		if b, err := uart[sercom].txBuffer.ReadByte(); err == nil {
			for !chip.SERCOM[sercom].USART_INT.INTFLAG.GetDRE() {
			}
			chip.SERCOM[sercom].USART_INT.DATA.SetDATA(uint32(b))
		} else {
			// Stop if there was an error reading the next byte
			break
		}
	}
	chip.SERCOM[sercom].USART_INT.INTENCLR.SetDRE(true)
}

func (u *UART) Read(p []byte) (n int, err error) {
	return u.rxBuffer.Read(p)
}

func (u *UART) Write(p []byte) (n int, err error) {
	u.mutex.Lock()
	// Write the string to the TX buffer
	n, err = u.txBuffer.Write(p)

	// Enable the DRE interrupt that will write the bytes from the buffer
	chip.SERCOM[u.sercom].USART_INT.INTENSET.SetDRE(true)
	u.mutex.Unlock()
	return
}

func (u *UART) WriteString(s string) (n int, err error) {
	u.mutex.Lock()
	// Write the string to the TX buffer
	n, err = u.txBuffer.WriteString(s)

	// Enable the DRE interrupt that will write the bytes from the buffer
	chip.SERCOM[u.sercom].USART_INT.INTENSET.SetDRE(true)
	u.mutex.Unlock()
	return
}
