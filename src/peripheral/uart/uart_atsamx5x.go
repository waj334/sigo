//go:build atsamx5x && !generic

package uart

import (
	"sync"

	"peripheral/pin"

	"runtime/arm/cortexm"
	"runtime/arm/cortexm/sam/atsamx5x"
	"runtime/arm/cortexm/sam/atsamx5x/support/mclk"
	usart "runtime/arm/cortexm/sam/atsamx5x/support/sercom/usartInt"
	"runtime/arm/cortexm/support/systemcontrol"
	"runtime/ringbuffer"
)

const (
	UsartFrame           = usart.CtrlaFormUsartFrameNoParity
	UsartFrameWithParity = usart.CtrlaFormUsartFrameWithParity
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
	atsamx5x.SERCOM
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
	FrameFormat     usart.ConstantSercomUsartCtrlaFormType
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
	usart.UsartInt[u.SERCOM].Ctrla.SetEnable(false)
	for usart.UsartInt[u.SERCOM].Syncbusy.GetEnable() {
	}

	// Enabled the SERCOM in MCLK
	switch u.SERCOM {
	case 0:
		mclk.Mclk.Apbamask.SetSercom0(true)
	case 1:
		mclk.Mclk.Apbamask.SetSercom1(true)
	case 2:
		mclk.Mclk.Apbbmask.SetSercom2(true)
	case 3:
		mclk.Mclk.Apbbmask.SetSercom3(true)
	case 4:
		mclk.Mclk.Apbdmask.SetSercom4(true)
	case 5:
		mclk.Mclk.Apbdmask.SetSercom5(true)
	case 6:
		mclk.Mclk.Apbdmask.SetSercom6(true)
	case 7:
		mclk.Mclk.Apbdmask.SetSercom7(true)
	}

	// Set the baud rate
	ratio := (uint64(atsamx5x.SercomRefFrequency) * uint64(1000)) / uint64(config.BaudHz*16)
	baud := ratio / 1000
	fp := ((ratio - (baud * 1000)) * 8) / 1000

	usart.UsartInt[u.SERCOM].Ctrla.SetSampr(usart.CtrlaSampr16XFractional)
	usart.UsartInt[u.SERCOM].Baud.SetFracfpBaud(uint16(baud))
	usart.UsartInt[u.SERCOM].Baud.SetFracfpFp(uint8(fp))

	usart.UsartInt[u.SERCOM].Ctrla.SetMode(usart.CtrlaModeUsartIntClk)
	usart.UsartInt[u.SERCOM].Ctrla.SetRxpo(usart.ConstantSercomUsartCtrlaRxpoType(rxPad))
	if config.XCK != 0 && config.RTS != 0 {
		usart.UsartInt[u.SERCOM].Ctrla.SetTxpo(usart.CtrlaTxpoPad3)
	} else if config.RTS != 0 && config.CTS != 0 {
		usart.UsartInt[u.SERCOM].Ctrla.SetTxpo(0x2)
	} else {
		usart.UsartInt[u.SERCOM].Ctrla.SetTxpo(usart.CtrlaTxpoPad0)
	}
	usart.UsartInt[u.SERCOM].Ctrla.SetDord(usart.CtrlaDordLsb)
	usart.UsartInt[u.SERCOM].Ctrla.SetCmode(usart.CtrlaCmodeAsync)

	switch config.CharacterSize {
	case 5:
		usart.UsartInt[u.SERCOM].Ctrlb.SetChsize(usart.CtrlbChsize5Bit)
	case 6:
		usart.UsartInt[u.SERCOM].Ctrlb.SetChsize(usart.CtrlbChsize6Bit)
	case 7:
		usart.UsartInt[u.SERCOM].Ctrlb.SetChsize(usart.CtrlbChsize7Bit)
	case 8:
		usart.UsartInt[u.SERCOM].Ctrlb.SetChsize(usart.CtrlbChsize8Bit)
	case 9:
		usart.UsartInt[u.SERCOM].Ctrlb.SetChsize(usart.CtrlbChsize9Bit)
	default:
		panic("invalid character size value")
	}
	for usart.UsartInt[u.SERCOM].Syncbusy.GetCtrlb() {
	}

	usart.UsartInt[u.SERCOM].Ctrla.SetForm(config.FrameFormat)

	switch config.NumStopBits {
	case 1:
		usart.UsartInt[u.SERCOM].Ctrlb.SetSbmode(usart.CtrlbSbmode1Bit)
	case 2:
		usart.UsartInt[u.SERCOM].Ctrlb.SetSbmode(usart.CtrlbSbmode2Bit)
	default:
		panic("invalid stop bits value")
	}
	for usart.UsartInt[u.SERCOM].Syncbusy.GetCtrlb() {
	}

	if config.ReceiveEnabled {
		usart.UsartInt[u.SERCOM].Ctrlb.SetRxen(true)
		for usart.UsartInt[u.SERCOM].Syncbusy.GetCtrlb() {
		}
	}

	if config.TransmitEnabled {
		usart.UsartInt[u.SERCOM].Ctrlb.SetTxen(true)
		for usart.UsartInt[u.SERCOM].Syncbusy.GetCtrlb() {
		}
	}

	rx := ringbuffer.New(config.RXBufferSize)
	tx := ringbuffer.New(config.TXBufferSize)

	u.rxBuffer = rx
	u.txBuffer = tx
	atsamx5x.SERCOMHandlers[u.SERCOM][0].Set(irqHandler)

	// Enable interrupts
	irqBase := 46 + u.SERCOM*4
	irq := cortexm.Interrupt(irqBase)
	irq.EnableIRQ()
	usart.UsartInt[u.SERCOM].Intenset.SetRxc(true)

	// Enable the peripheral
	usart.UsartInt[u.SERCOM].Ctrla.SetEnable(true)
	for usart.UsartInt[u.SERCOM].Syncbusy.GetEnable() {
	}
}

func irqHandler() {
	sercom := int(systemcontrol.SystemControl.Icsr.GetVectactive()-62) / 4
	switch {
	case usart.UsartInt[sercom].Intflag.GetRxc():
		rxcHandler(sercom)
	case usart.UsartInt[sercom].Intflag.GetDre():
		dreHandler(sercom)
	}
}

func rxcHandler(sercom int) {
	b := byte(usart.UsartInt[sercom].Data.GetData())
	uart[sercom].rxBuffer.WriteByte(b)
}

func dreHandler(sercom int) {
	for uart[sercom].txBuffer.Len() > 0 {
		if b, err := uart[sercom].txBuffer.ReadByte(); err == nil {
			for !usart.UsartInt[sercom].Intflag.GetDre() {
			}
			usart.UsartInt[sercom].Data.SetData(uint32(b))
		} else {
			// Stop if there was an error reading the next byte
			break
		}
	}
	usart.UsartInt[sercom].Intenclr.SetDre(true)
}

func (u *UART) Read(p []byte) (n int, err error) {
	return u.rxBuffer.Read(p)
}

func (u *UART) Write(p []byte) (n int, err error) {
	u.mutex.Lock()
	// Write the string to the _TX buffer
	n, err = u.txBuffer.Write(p)

	// Enable the DRE interrupt that will write the bytes from the buffer
	usart.UsartInt[u.SERCOM].Intenset.SetDre(true)
	u.mutex.Unlock()
	return
}

func (u *UART) WriteString(s string) (n int, err error) {
	u.mutex.Lock()
	// Write the string to the _TX buffer
	n, err = u.txBuffer.WriteString(s)

	// Enable the DRE interrupt that will write the bytes from the buffer
	usart.UsartInt[u.SERCOM].Intenset.SetDre(true)
	u.mutex.Unlock()
	return
}
