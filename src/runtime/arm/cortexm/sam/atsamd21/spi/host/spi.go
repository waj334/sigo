package host

import (
	"runtime/arm/cortexm/sam/atsamd21"
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
	spi  = [6]*SPI{
		SPI0,
		SPI1,
		SPI2,
		SPI3,
		SPI4,
		SPI5,
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

	errStrInvalidPin = "(SPI): invalid pin configuration"
)

type SPI struct {
	atsamd21.SERCOM
	config   Config
	txBuffer ringbuffer.RingBuffer
	rxBuffer ringbuffer.RingBuffer
	mutex    sync.Mutex
}

type Config struct {
	DI  atsamd21.Pin
	DO  atsamd21.Pin
	SCK atsamd21.Pin
	CS  atsamd21.Pin

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

func spiValidateDIPO(do, sck, cs int, mssen bool) (dopo chip.SERCOM_SPIM_CTRLA_REG_DOPO, ok bool) {
	ok = true
	if do == 0 {
		if sck == 1 {
			dopo = chip.SERCOM_SPIM_CTRLA_REG_DOPO_PAD0
		} else if sck != 3 {
			dopo = chip.SERCOM_SPIM_CTRLA_REG_DOPO_PAD3
		} else {
			return 0, false
		}

		if mssen {
			if cs != 2 && cs != 1 {
				return 0, false
			}
		}
	} else if do == 2 {
		if sck == 3 {
			dopo = chip.SERCOM_SPIM_CTRLA_REG_DOPO_PAD1
		} else {
			return 0, false
		}

		if mssen && cs != 1 {
			return 0, false
		}
	} else if do == 3 {
		if sck == 1 {
			dopo = chip.SERCOM_SPIM_CTRLA_REG_DOPO_PAD2
		} else {
			return 0, false
		}

		if mssen && cs != 2 {
			return 0, false
		}
	}
	return
}

func (s *SPI) Configure(config Config) {
	var mode atsamd21.PMUXFunction
	var doPad chip.SERCOM_SPIM_CTRLA_REG_DOPO
	var diPad chip.SERCOM_SPIM_CTRLA_REG_DIPO
	alt := false

	// Validate pinout
	if dopo, ok := spiValidateDIPO(config.DO.GetPAD(), config.SCK.GetPAD(), config.CS.GetPAD(), config.HardwareSelect); ok {
		doPad = dopo
	} else if dopo, ok = spiValidateDIPO(config.DO.GetAltPAD(), config.SCK.GetAltPAD(), config.CS.GetAltPAD(), config.HardwareSelect); ok {
		alt = true
		doPad = dopo
	} else {
		panic(errStrInvalidPin)
	}

	if config.DI == atsamd21.NoPin ||
		config.DI == config.CS ||
		config.DI == config.DO ||
		config.DI == config.SCK {
		panic(errStrInvalidPin)
	} else {
		pad := config.DI.GetPAD()
		if alt {
			pad = config.DI.GetAltPAD()
		}
		switch pad {
		case 0:
			diPad = chip.SERCOM_SPIM_CTRLA_REG_DIPO_PAD0
		case 1:
			diPad = chip.SERCOM_SPIM_CTRLA_REG_DIPO_PAD1
		case 2:
			diPad = chip.SERCOM_SPIM_CTRLA_REG_DIPO_PAD2
		case 3:
			diPad = chip.SERCOM_SPIM_CTRLA_REG_DIPO_PAD3
		default:
			panic(errStrInvalidPin)
		}
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

	// Reset the SERCOM
	chip.SERCOM_SPIM[s.SERCOM].CTRLA.SetSWRST(true)
	s.Synchronize()

	// Enable the SERCOM
	s.SetPMEnabled(true)

	// Calculate the BAUD value
	baud := s.Baud(config.BaudHz)

	// Set up the registers
	chip.SERCOM_SPIM[s.SERCOM].BAUD.SetBAUD(baud)

	chip.SERCOM_SPIM[s.SERCOM].CTRLA.SetDORD(config.DataOrder)
	chip.SERCOM_SPIM[s.SERCOM].CTRLA.SetFORM(config.Form)
	chip.SERCOM_SPIM[s.SERCOM].CTRLA.SetMODE(config.Mode)
	chip.SERCOM_SPIM[s.SERCOM].CTRLA.SetCPHA(config.Phase)
	chip.SERCOM_SPIM[s.SERCOM].CTRLA.SetCPOL(config.Polarity)
	chip.SERCOM_SPIM[s.SERCOM].CTRLA.SetDIPO(diPad)
	chip.SERCOM_SPIM[s.SERCOM].CTRLA.SetDOPO(doPad)
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
	s.SetHandler(irqHandler)

	// Enable interrupts
	s.Irq().EnableIRQ()
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
	sercom := int(chip.SystemControl.ICSR.GetVECTACTIVE()-16) - int(atsamd21.IRQ_SERCOM0)
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
			chip.SERCOM_SPIM[sercom].DATA.SetDATA(uint16(b))
		} else {
			// Stop if there was an error reading the next byte
			break
		}
	}
	chip.SERCOM_SPIM[sercom].INTENCLR.SetDRE(true)
}
