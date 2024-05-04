//go:build samx51 && !generic

package uart

import (
	"peripheral/pin"
	"runtime/arm/cortexm"
	"runtime/arm/cortexm/sam/chip"
	"runtime/arm/cortexm/sam/samx51"
	"runtime/ringbuffer"
	"sync"
)

const (
	UsartFrame           = chip.SERCOM_USART_INT_CTRLA_REG_FORM_USART_FRAME_NO_PARITY
	UsartFrameWithParity = chip.SERCOM_USART_INT_CTRLA_REG_FORM_USART_FRAME_WITH_PARITY
)

var (
	UART0 = &UART{SERCOM: 0}
	UART1 = &UART{SERCOM: 1}
	UART2 = &UART{SERCOM: 2}
	UART3 = &UART{SERCOM: 3}
	UART4 = &UART{SERCOM: 4}
	UART5 = &UART{SERCOM: 5}
	UART6 = &UART{SERCOM: 6}
	UART7 = &UART{SERCOM: 7}
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
	samx51.SERCOM
	txBuffer ringbuffer.RingBuffer
	rxBuffer ringbuffer.RingBuffer
	mutex    sync.Mutex
}

type Config struct {
	TXD             pin.Pin
	RXD             pin.Pin
	XCK             pin.Pin
	RTS             pin.Pin
	CTS             pin.Pin
	FrameFormat     chip.SERCOM_USART_INT_CTRLA_REG_FORM
	BaudHz          uint
	CharacterSize   uint
	NumStopBits     uint
	ReceiveEnabled  bool
	TransmitEnabled bool
	RXBufferSize    uintptr
	TXBufferSize    uintptr
}

func (u *UART) Configure(config Config) {
	var mode pin.PMUXFunction
	var rxPad int

	// Determine the SERCOM number from the PAD value of TXD pin
	if config.TXD.GetPAD() == 0 {
		mode = pin.PMUXFunctionC
		rxPad = config.RXD.GetPAD()
		// Check the other optional pins
		if config.TXD.GetSERCOM() != u.SERCOM ||
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
		mode = pin.PMUXFunctionD
		rxPad = config.RXD.GetAltPAD()
		// Check the other optional pins
		if config.TXD.GetAltSERCOM() != u.SERCOM ||
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
	chip.SERCOM_USART_INT[u.SERCOM].CTRLA.SetENABLE(false)
	for chip.SERCOM_USART_INT[u.SERCOM].SYNCBUSY.GetENABLE() {
	}

	// Enabled the SERCOM in MCLK
	switch u.SERCOM {
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
	ratio := (uint64(samx51.SERCOM_REF_FREQUENCY) * uint64(1000)) / uint64(config.BaudHz*16)
	baud := ratio / 1000
	fp := ((ratio - (baud * 1000)) * 8) / 1000

	chip.SERCOM_USART_INT[u.SERCOM].CTRLA.SetSAMPR(chip.SERCOM_USART_INT_CTRLA_REG_SAMPR_16X_FRACTIONAL)
	chip.SERCOM_USART_INT[u.SERCOM].BAUD.SetFRACFPBAUD(uint16(baud))
	chip.SERCOM_USART_INT[u.SERCOM].BAUD.SetFRACFPFP(uint8(fp))

	chip.SERCOM_USART_INT[u.SERCOM].CTRLA.SetMODE(chip.SERCOM_USART_INT_CTRLA_REG_MODE_USART_INT_CLK)
	chip.SERCOM_USART_INT[u.SERCOM].CTRLA.SetRXPO(chip.SERCOM_USART_INT_CTRLA_REG_RXPO(rxPad))
	if config.XCK != 0 && config.RTS != 0 {
		chip.SERCOM_USART_INT[u.SERCOM].CTRLA.SetTXPO(chip.SERCOM_USART_INT_CTRLA_REG_TXPO_PAD3)
	} else if config.RTS != 0 && config.CTS != 0 {
		chip.SERCOM_USART_INT[u.SERCOM].CTRLA.SetTXPO(chip.SERCOM_USART_INT_CTRLA_REG_TXPO_PAD2)
	} else {
		chip.SERCOM_USART_INT[u.SERCOM].CTRLA.SetTXPO(chip.SERCOM_USART_INT_CTRLA_REG_TXPO_PAD0)
	}
	chip.SERCOM_USART_INT[u.SERCOM].CTRLA.SetDORD(chip.SERCOM_USART_INT_CTRLA_REG_DORD_LSB)
	chip.SERCOM_USART_INT[u.SERCOM].CTRLA.SetCMODE(chip.SERCOM_USART_INT_CTRLA_REG_CMODE_ASYNC)

	switch config.CharacterSize {
	case 5:
		chip.SERCOM_USART_INT[u.SERCOM].CTRLB.SetCHSIZE(chip.SERCOM_USART_INT_CTRLB_REG_CHSIZE_5_BIT)
	case 6:
		chip.SERCOM_USART_INT[u.SERCOM].CTRLB.SetCHSIZE(chip.SERCOM_USART_INT_CTRLB_REG_CHSIZE_6_BIT)
	case 7:
		chip.SERCOM_USART_INT[u.SERCOM].CTRLB.SetCHSIZE(chip.SERCOM_USART_INT_CTRLB_REG_CHSIZE_7_BIT)
	case 8:
		chip.SERCOM_USART_INT[u.SERCOM].CTRLB.SetCHSIZE(chip.SERCOM_USART_INT_CTRLB_REG_CHSIZE_8_BIT)
	case 9:
		chip.SERCOM_USART_INT[u.SERCOM].CTRLB.SetCHSIZE(chip.SERCOM_USART_INT_CTRLB_REG_CHSIZE_9_BIT)
	default:
		panic("invalid character size value")
	}
	for chip.SERCOM_USART_INT[u.SERCOM].SYNCBUSY.GetCTRLB() {
	}

	chip.SERCOM_USART_INT[u.SERCOM].CTRLA.SetFORM(config.FrameFormat)

	switch config.NumStopBits {
	case 1:
		chip.SERCOM_USART_INT[u.SERCOM].CTRLB.SetSBMODE(chip.SERCOM_USART_INT_CTRLB_REG_SBMODE_1_BIT)
	case 2:
		chip.SERCOM_USART_INT[u.SERCOM].CTRLB.SetSBMODE(chip.SERCOM_USART_INT_CTRLB_REG_SBMODE_2_BIT)
	default:
		panic("invalid stop bits value")
	}
	for chip.SERCOM_USART_INT[u.SERCOM].SYNCBUSY.GetCTRLB() {
	}

	if config.ReceiveEnabled {
		chip.SERCOM_USART_INT[u.SERCOM].CTRLB.SetRXEN(true)
		for chip.SERCOM_USART_INT[u.SERCOM].SYNCBUSY.GetCTRLB() {
		}
	}

	if config.TransmitEnabled {
		chip.SERCOM_USART_INT[u.SERCOM].CTRLB.SetTXEN(true)
		for chip.SERCOM_USART_INT[u.SERCOM].SYNCBUSY.GetCTRLB() {
		}
	}

	rx := ringbuffer.New(config.RXBufferSize)
	tx := ringbuffer.New(config.TXBufferSize)

	u.rxBuffer = rx
	u.txBuffer = tx
	samx51.SERCOMHandlers[u.SERCOM][0].Set(irqHandler)

	// Enable interrupts
	irqBase := 46 + u.SERCOM*4
	irq := cortexm.Interrupt(irqBase)
	irq.EnableIRQ()
	chip.SERCOM_USART_INT[u.SERCOM].INTENSET.SetRXC(true)

	// Enable the peripheral
	chip.SERCOM_USART_INT[u.SERCOM].CTRLA.SetENABLE(true)
	for chip.SERCOM_USART_INT[u.SERCOM].SYNCBUSY.GetENABLE() {
	}
}

func irqHandler() {
	sercom := int(chip.SystemControl.ICSR.GetVECTACTIVE()-62) / 4
	switch {
	case chip.SERCOM_USART_INT[sercom].INTFLAG.GetRXC():
		rxcHandler(sercom)
	case chip.SERCOM_USART_INT[sercom].INTFLAG.GetDRE():
		dreHandler(sercom)
	}
}

func rxcHandler(sercom int) {
	b := byte(chip.SERCOM_USART_INT[sercom].DATA.GetDATA())
	uart[sercom].rxBuffer.WriteByte(b)
}

func dreHandler(sercom int) {
	for uart[sercom].txBuffer.Len() > 0 {
		if b, err := uart[sercom].txBuffer.ReadByte(); err == nil {
			for !chip.SERCOM_USART_INT[sercom].INTFLAG.GetDRE() {
			}
			chip.SERCOM_USART_INT[sercom].DATA.SetDATA(uint32(b))
		} else {
			// Stop if there was an error reading the next byte
			break
		}
	}
	chip.SERCOM_USART_INT[sercom].INTENCLR.SetDRE(true)
}

func (u *UART) Read(p []byte) (n int, err error) {
	return u.rxBuffer.Read(p)
}

func (u *UART) Write(p []byte) (n int, err error) {
	u.mutex.Lock()
	// Write the string to the TX buffer
	n, err = u.txBuffer.Write(p)

	// Enable the DRE interrupt that will write the bytes from the buffer
	chip.SERCOM_USART_INT[u.SERCOM].INTENSET.SetDRE(true)
	u.mutex.Unlock()
	return
}

func (u *UART) WriteString(s string) (n int, err error) {
	u.mutex.Lock()
	// Write the string to the TX buffer
	n, err = u.txBuffer.WriteString(s)

	// Enable the DRE interrupt that will write the bytes from the buffer
	chip.SERCOM_USART_INT[u.SERCOM].INTENSET.SetDRE(true)
	u.mutex.Unlock()
	return
}