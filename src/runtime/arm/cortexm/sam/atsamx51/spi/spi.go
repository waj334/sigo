package spi

import (
	"runtime/arm/cortexm/sam/atsamx51"
	"runtime/arm/cortexm/sam/chip"
	"runtime/ringbuffer"
	"sync"
)

var (
	SPI0 = &SPI{SERCOM: 0}
	SPI1 = &SPI{SERCOM: 1}
	SPI2 = &SPI{SERCOM: 2}
	SPI3 = &SPI{SERCOM: 3}
	SPI4 = &SPI{SERCOM: 4}
	SPI5 = &SPI{SERCOM: 5}
	SPI6 = &SPI{SERCOM: 6}
	SPI7 = &SPI{SERCOM: 7}
	spi  = [8]*SPI{
		SPI0,
		SPI1,
		SPI2,
		SPI3,
		SPI4,
		SPI5,
		SPI6,
		SPI7,
	}
)

const (
	HostMode   = chip.SERCOM_SPIM_CTRLA_REG_MODE_SPI_MASTER
	ClientMode = chip.SERCOM_SPIM_CTRLA_REG_MODE_SPI_SLAVE

	MSB = chip.SERCOM_SPIM_CTRLA_REG_DORD_MSB
	LSB = chip.SERCOM_SPIM_CTRLA_REG_DORD_LSB

	Frame            = chip.SERCOM_SPIM_CTRLA_REG_FORM_SPI_FRAME
	FrameWithAddress = chip.SERCOM_SPIM_CTRLA_REG_FORM_SPI_FRAME_WITH_ADDR

	LeadingEdge  = chip.SERCOM_SPIM_CTRLA_REG_CPHA_LEADING_EDGE
	TrailingEdge = chip.SERCOM_SPIM_CTRLA_REG_CPHA_TRAILING_EDGE

	IdleLow  = chip.SERCOM_SPIM_CTRLA_REG_CPOL_IDLE_LOW
	IdleHigh = chip.SERCOM_SPIM_CTRLA_REG_CPOL_IDLE_HIGH
)

const (
	errStrInvalidPin = "(SPI): invalid pin configuration"
)

type SPI struct {
	atsamx51.SERCOM
	config   Config
	txBuffer ringbuffer.RingBuffer
	rxBuffer ringbuffer.RingBuffer
	mutex    sync.Mutex
}

type Config struct {
	DI  atsamx51.Pin
	DO  atsamx51.Pin
	SCK atsamx51.Pin
	CS  atsamx51.Pin

	BaudHz         uint
	CharacterSize  uint8
	DataOrder      chip.SERCOM_SPIM_CTRLA_REG_DORD
	Form           chip.SERCOM_SPIM_CTRLA_REG_FORM
	HardwareSelect bool
	Mode           chip.SERCOM_SPIM_CTRLA_REG_MODE
	Phase          chip.SERCOM_SPIM_CTRLA_REG_CPHA
	Polarity       chip.SERCOM_SPIM_CTRLA_REG_CPOL
	ReceiveEnabled bool

	RXBufferSize uintptr
	TXBufferSize uintptr
}

func (s *SPI) Configure(config Config) {
	var mode atsamx51.PMUXFunction
	var doPad int
	var diPad int

	// Validate pinout
	if config.DO.GetPAD() == 0 || config.DO.GetPAD() == 3 {
		mode = atsamx51.PMUXFunctionC
		doPad = config.DO.GetPAD()
		diPad = config.DI.GetPAD()
		if diPad == config.DO.GetPAD() ||
			diPad == config.SCK.GetPAD() ||
			(config.HardwareSelect && diPad == config.CS.GetPAD()) ||
			config.SCK.GetPAD() != 1 ||
			(config.HardwareSelect && config.CS.GetPAD() != 2) ||
			config.DO == atsamx51.NoPin ||
			config.DI == atsamx51.NoPin ||
			config.SCK == atsamx51.NoPin ||
			(config.HardwareSelect && config.CS == atsamx51.NoPin) {
			panic(errStrInvalidPin)
		}
	} else if config.DO.GetAltPAD() == 0 || config.DO.GetAltPAD() == 3 {
		mode = atsamx51.PMUXFunctionD
		doPad = config.DO.GetAltPAD()
		diPad = config.DI.GetAltPAD()
		if diPad == config.DO.GetAltPAD() ||
			diPad == config.SCK.GetAltPAD() ||
			(config.HardwareSelect && diPad == config.CS.GetAltPAD()) ||
			config.SCK.GetAltPAD() != 1 ||
			(config.HardwareSelect && config.CS.GetAltPAD() != 2) ||
			config.DO == atsamx51.NoPin ||
			config.DI == atsamx51.NoPin ||
			config.SCK == atsamx51.NoPin ||
			(config.HardwareSelect && config.CS == atsamx51.NoPin) {
			panic(errStrInvalidPin)
		}
	} else {
		panic(errStrInvalidPin)
	}

	// DO can be on either PAD0 (0x0) or PAD3 (0x2)
	if doPad == 3 {
		// Set to the alternate pinout
		doPad = 2
	}

	// Set the pin configurations
	config.DI.SetPMUX(mode, true)
	config.DO.SetPMUX(mode, true)
	config.SCK.SetPMUX(mode, true)
	if config.HardwareSelect {
		config.CS.SetPMUX(mode, true)
	} else {
		// Set the CS pin to output mode
		config.CS.SetDirection(1)
	}

	// Enable the SERCOM
	s.SetEnabled(true)

	// Calculate the BAUD value
	baud := s.Baud(config.BaudHz)

	// Set up the registers
	chip.SERCOM_SPIM[s.SERCOM].BAUD.SetBAUD(baud)

	chip.SERCOM_SPIM[s.SERCOM].CTRLA.SetDORD(config.DataOrder)
	chip.SERCOM_SPIM[s.SERCOM].CTRLA.SetFORM(config.Form)
	chip.SERCOM_SPIM[s.SERCOM].CTRLA.SetMODE(config.Mode)
	chip.SERCOM_SPIM[s.SERCOM].CTRLA.SetCPHA(config.Phase)
	chip.SERCOM_SPIM[s.SERCOM].CTRLA.SetCPOL(config.Polarity)
	chip.SERCOM_SPIM[s.SERCOM].CTRLA.SetDIPO(chip.SERCOM_SPIM_CTRLA_REG_DIPO(diPad))
	chip.SERCOM_SPIM[s.SERCOM].CTRLA.SetDOPO(chip.SERCOM_SPIM_CTRLA_REG_DOPO(doPad))
	chip.SERCOM_SPIM[s.SERCOM].CTRLA.SetCPOL(config.Polarity)

	chip.SERCOM_SPIM[s.SERCOM].CTRLB.SetRXEN(config.ReceiveEnabled)
	chip.SERCOM_SPIM[s.SERCOM].CTRLB.SetMSSEN(config.HardwareSelect)
	switch config.CharacterSize {
	case 9:
		chip.SERCOM_SPIM[s.SERCOM].CTRLB.SetCHSIZE(chip.SERCOM_SPIM_CTRLB_REG_CHSIZE_9_BIT)
	default:
		chip.SERCOM_SPIM[s.SERCOM].CTRLB.SetCHSIZE(chip.SERCOM_SPIM_CTRLB_REG_CHSIZE_8_BIT)
	}

	// Set up buffers
	rx := ringbuffer.New(config.RXBufferSize)
	tx := ringbuffer.New(config.TXBufferSize)

	s.rxBuffer = rx
	s.txBuffer = tx

	// Set the interrupt handler
	atsamx51.SERCOMHandlers[s.SERCOM][0].Set(irqHandler)

	// Enable interrupts
	s.Irq0().EnableIRQ()
	chip.SERCOM_SPIM[s.SERCOM].INTENSET.SetRXC(true)

	// Enable the peripheral
	chip.SERCOM_SPIM[s.SERCOM].CTRLA.SetENABLE(true)
	for chip.SERCOM_SPIM[s.SERCOM].SYNCBUSY.GetENABLE() {
	}

	s.config = config
}

func (s *SPI) Read(p []byte) (n int, err error) {
	return s.rxBuffer.Read(p)
}

func (s *SPI) Write(p []byte) (n int, err error) {
	s.mutex.Lock()
	// Write the string to the TX buffer
	n, err = s.txBuffer.Write(p)

	// Enable the DRE interrupt that will write the bytes from the buffer
	chip.SERCOM_SPIM[s.SERCOM].INTENSET.SetDRE(true)
	s.mutex.Unlock()
	return
}

func (s *SPI) Transact(b []byte) []byte {
	//TODO implement me
	panic("implement me")
}

func (s *SPI) Select() {
	if !s.config.HardwareSelect {
		s.config.CS.Set(false)
	}
}

func (s *SPI) Deselect() {
	if !s.config.HardwareSelect {
		s.config.CS.Set(true)
	}
}

func irqHandler() {
	sercom := int(chip.SystemControl.ICSR.GetVECTACTIVE()-62) / 4
	switch {
	case chip.SERCOM_SPIM[sercom].INTFLAG.GetRXC():
		rxcHandler(sercom)
	case chip.SERCOM_SPIM[sercom].INTFLAG.GetDRE():
		dreHandler(sercom)
	}
}

func rxcHandler(sercom int) {
	b := byte(chip.SERCOM_SPIM[sercom].DATA.GetDATA())
	spi[sercom].rxBuffer.WriteByte(b)
}

func dreHandler(sercom int) {
	for spi[sercom].txBuffer.Len() > 0 {
		if b, err := spi[sercom].txBuffer.ReadByte(); err == nil {
			for !chip.SERCOM_SPIM[sercom].INTFLAG.GetDRE() {
			}
			chip.SERCOM_SPIM[sercom].DATA.SetDATA(uint32(b))
		} else {
			// Stop if there was an error reading the next byte
			break
		}
	}
	chip.SERCOM_SPIM[sercom].INTENCLR.SetDRE(true)
}
