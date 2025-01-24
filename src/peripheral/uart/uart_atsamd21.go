//go:build atsamd21 && !generic

package uart

import (
	"sync"

	"peripheral/pin"

	"runtime/arm/cortexm/sam/atsamd21"
	"runtime/arm/cortexm/sam/atsamd21/support/sercom/usartInt"
	"runtime/arm/cortexm/support/systemcontrol"
	"runtime/ringbuffer"
)

const (
	UsartFrame           = usartInt.CtrlaFormUsartFrameNoParity
	UsartFrameWithParity = usartInt.CtrlaFormUsartFrameWithParity
)

var (
	UART0 = &_uart{SERCOM: 0}
	UART1 = &_uart{SERCOM: 1}
	UART2 = &_uart{SERCOM: 2}
	UART3 = &_uart{SERCOM: 3}
	UART4 = &_uart{SERCOM: 4}
	UART5 = &_uart{SERCOM: 5}
	uart  = [6]*_uart{
		UART0,
		UART1,
		UART2,
		UART3,
		UART4,
		UART5,
	}
)

type _uart struct {
	atsamd21.SERCOM
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
	FrameFormat     usartInt.ConstantSercomUsartCtrlaFormType
	BaudHz          uint
	CharacterSize   uint
	NumStopBits     uint
	ReceiveEnabled  bool
	TransmitEnabled bool
	RXBufferSize    uintptr
	TXBufferSize    uintptr
}

func (u *_uart) Configure(config Config) {
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
	} else if config.TXD.GetPAD() == 2 {
		mode = pin.PMUXFunctionC
		rxPad = config.RXD.GetPAD()
		// Check the other optional pins
		if config.TXD.GetSERCOM() != u.SERCOM ||
			config.XCK != 0 && config.XCK.GetPAD() != 3 ||
			config.RTS != 0 && config.RTS.GetPAD() != 0xFF || // NoPAD
			config.CTS != 0 && config.CTS.GetPAD() != 0xFF || // NoPAD
			// The receive pad must not conflict with any of the other pads
			rxPad == config.TXD.GetPAD() ||
			rxPad == config.XCK.GetPAD() ||
			rxPad == config.RTS.GetPAD() ||
			rxPad == config.CTS.GetPAD() {
			panic("invalid selection")
		}
	} else if config.TXD.GetAltPAD() == 2 {
		mode = pin.PMUXFunctionD
		rxPad = config.RXD.GetAltPAD()
		// Check the other optional pins
		if config.TXD.GetAltSERCOM() != u.SERCOM ||
			config.XCK != 0 && config.XCK.GetAltPAD() != 3 ||
			config.RTS != 0 && config.RTS.GetAltPAD() != 0xFF || // NoPAD
			config.CTS != 0 && config.CTS.GetAltPAD() != 0xFF || // NoPAD
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

	// Reset the SERCOM
	usartInt.UsartInt[u.SERCOM].Ctrla.SetSwrst(true)
	u.Synchronize()

	// Enable the SERCOM in PM
	u.SERCOM.SetPMEnabled(true)

	// Set the baud rate
	ratio := (uint64(atsamd21.SercomRefFrequency) * uint64(1000)) / uint64(config.BaudHz*16)
	baud := ratio / 1000
	fp := ((ratio - (baud * 1000)) * 8) / 1000

	usartInt.UsartInt[u.SERCOM].Ctrla.SetSampr(usartInt.CtrlaSampr16XFractional)
	usartInt.UsartInt[u.SERCOM].Baud.SetFracfpBaud(uint16(baud))
	usartInt.UsartInt[u.SERCOM].Baud.SetFracfpFp(uint8(fp))

	usartInt.UsartInt[u.SERCOM].Ctrla.SetMode(usartInt.CtrlaModeUsartIntClk)
	usartInt.UsartInt[u.SERCOM].Ctrla.SetRxpo(usartInt.ConstantSercomUsartCtrlaRxpoType(rxPad))
	if config.XCK != 0 && config.RTS != 0 {
		panic("configuration not available for SAMD21")
	} else if config.RTS != 0 && config.CTS != 0 {
		usartInt.UsartInt[u.SERCOM].Ctrla.SetTxpo(usartInt.CtrlaTxpoPad2)
	} else if config.TXD.GetPAD() == 2 || config.TXD.GetAltPAD() == 2 {
		usartInt.UsartInt[u.SERCOM].Ctrla.SetTxpo(usartInt.CtrlaTxpoPad1)
	} else {
		usartInt.UsartInt[u.SERCOM].Ctrla.SetTxpo(usartInt.CtrlaTxpoPad0)
	}
	usartInt.UsartInt[u.SERCOM].Ctrla.SetDord(usartInt.CtrlaDordLsb)
	usartInt.UsartInt[u.SERCOM].Ctrla.SetCmode(usartInt.CtrlaCmodeAsync)

	switch config.CharacterSize {
	case 5:
		usartInt.UsartInt[u.SERCOM].Ctrlb.SetChsize(usartInt.CtrlbChsize5Bit)
	case 6:
		usartInt.UsartInt[u.SERCOM].Ctrlb.SetChsize(usartInt.CtrlbChsize6Bit)
	case 7:
		usartInt.UsartInt[u.SERCOM].Ctrlb.SetChsize(usartInt.CtrlbChsize7Bit)
	case 8:
		usartInt.UsartInt[u.SERCOM].Ctrlb.SetChsize(usartInt.CtrlbChsize8Bit)
	case 9:
		usartInt.UsartInt[u.SERCOM].Ctrlb.SetChsize(usartInt.CtrlbChsize9Bit)
	default:
		panic("invalid character size value")
	}
	for usartInt.UsartInt[u.SERCOM].Syncbusy.GetCtrlb() {
	}

	usartInt.UsartInt[u.SERCOM].Ctrla.SetForm(config.FrameFormat)

	switch config.NumStopBits {
	case 1:
		usartInt.UsartInt[u.SERCOM].Ctrlb.SetSbmode(usartInt.CtrlbSbmode1Bit)
	case 2:
		usartInt.UsartInt[u.SERCOM].Ctrlb.SetSbmode(usartInt.CtrlbSbmode2Bit)
	default:
		panic("invalid stop bits value")
	}
	for usartInt.UsartInt[u.SERCOM].Syncbusy.GetCtrlb() {
	}

	if config.ReceiveEnabled {
		usartInt.UsartInt[u.SERCOM].Ctrlb.SetRxen(true)
		for usartInt.UsartInt[u.SERCOM].Syncbusy.GetCtrlb() {
		}
	}

	if config.TransmitEnabled {
		usartInt.UsartInt[u.SERCOM].Ctrlb.SetTxen(true)
		for usartInt.UsartInt[u.SERCOM].Syncbusy.GetCtrlb() {
		}
	}

	rx := ringbuffer.New(config.RXBufferSize)
	tx := ringbuffer.New(config.TXBufferSize)

	u.rxBuffer = rx
	u.txBuffer = tx

	// Set the interrupt handler function
	u.SERCOM.SetHandler(irqHandler)

	// Enable interrupts
	u.SERCOM.Irq().EnableIRQ()
	usartInt.UsartInt[u.SERCOM].Intenset.SetRxc(true)

	// Enable the peripheral
	usartInt.UsartInt[u.SERCOM].Ctrla.SetEnable(true)
	for usartInt.UsartInt[u.SERCOM].Syncbusy.GetEnable() {
	}
}

func irqHandler() {
	sercom := int(systemcontrol.SystemControl.Icsr.GetVectactive()-16) - int(atsamd21.IRQ_SERCOM0)
	switch {
	case usartInt.UsartInt[sercom].Intflag.GetRxc():
		rxcHandler(sercom)
	case usartInt.UsartInt[sercom].Intflag.GetDre():
		dreHandler(sercom)
	}
}

func rxcHandler(sercom int) {
	b := byte(usartInt.UsartInt[sercom].Data.GetData())
	uart[sercom].rxBuffer.WriteByte(b)
}

func dreHandler(sercom int) {
	for uart[sercom].txBuffer.Len() > 0 {
		if b, err := uart[sercom].txBuffer.ReadByte(); err == nil {
			for !usartInt.UsartInt[sercom].Intflag.GetDre() {
			}
			usartInt.UsartInt[sercom].Data.SetData(uint16(b) & 0x1FF)
		} else {
			// Stop if there was an error reading the next byte
			break
		}
	}
	usartInt.UsartInt[sercom].Intenclr.SetDre(true)
}

func (u *_uart) Read(p []byte) (n int, err error) {
	return u.rxBuffer.Read(p)
}

func (u *_uart) Write(p []byte) (n int, err error) {
	u.mutex.Lock()
	// Write the string to the _TX buffer
	n, err = u.txBuffer.Write(p)

	// Enable the DRE interrupt that will write the bytes from the buffer
	usartInt.UsartInt[u.SERCOM].Intenset.SetDre(true)
	u.mutex.Unlock()
	return
}

func (u *_uart) WriteString(s string) (n int, err error) {
	u.mutex.Lock()
	// Write the string to the _TX buffer
	n, err = u.txBuffer.WriteString(s)

	// Enable the DRE interrupt that will write the bytes from the buffer
	usartInt.UsartInt[u.SERCOM].Intenset.SetDre(true)
	u.mutex.Unlock()
	return
}
