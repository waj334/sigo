//go:build samd21 && !generic

package spi

import (
	"peripheral"
	"peripheral/pin"
	"runtime/arm/cortexm/sam/chip"
	"runtime/arm/cortexm/sam/samd21"
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

const (
	stateIdle uint8 = iota
	stateReading
	stateWriting
	stateTransacting
	stateDone
)

type SPI struct {
	samd21.SERCOM
	cs         pin.Pin
	hardwareCS bool
	mutex      sync.Mutex

	rxBuffer []byte
	txBuffer []byte
	busMutex sync.Mutex
	nbytes   int
	state    uint8
}

type Config struct {
	DI  pin.Pin
	DO  pin.Pin
	SCK pin.Pin
	CS  pin.Pin

	BaudHz         uint
	CharacterSize  uint8
	DataOrder      chip.SERCOM_SPIM_CTRLA_REG_DORD
	Form           chip.SERCOM_SPIM_CTRLA_REG_FORM
	HardwareSelect bool
	Mode           chip.SERCOM_SPIM_CTRLA_REG_MODE
	Phase          chip.SERCOM_SPIM_CTRLA_REG_CPHA
	Polarity       chip.SERCOM_SPIM_CTRLA_REG_CPOL
	ReceiveEnabled bool
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
	var mode pin.PMUXFunction
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

	if config.DI == pin.NoPin ||
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

	// Set the interrupt handler
	s.SetHandler(irqHandler)

	// Enable interrupts
	s.Irq().EnableIRQ()

	// Enable the peripheral
	chip.SERCOM_SPIM[s.SERCOM].CTRLA.SetENABLE(true)
	for chip.SERCOM_SPIM[s.SERCOM].SYNCBUSY.GetENABLE() {
	}

	s.cs = config.CS
	s.hardwareCS = config.HardwareSelect

	// Lock the bus mutex
	s.busMutex.Lock()
}

func (s *SPI) Read(p []byte) (n int, err error) {
	s.mutex.Lock()

	// Set up the transaction
	s.rxBuffer = p
	s.state = stateReading

	// Enable receive interrupt so the incoming data can be written to the RX buffer
	chip.SERCOM_SPIM[s.SERCOM].INTENSET.SetRXC(true)

	// Write a byte to the DATA register to begin the transactions
	chip.SERCOM_SPIM[s.SERCOM].DATA.SetDATA(0)

	s.busMutex.Lock()

	// ...
	// Wait for transactions to complete
	// ...

	// Disable the receive interrupt since the RX buffer will be unset
	chip.SERCOM_SPIM[s.SERCOM].INTENSET.SetRXC(false)

	// Reset the state
	s.state = stateIdle
	s.rxBuffer = nil

	n = s.nbytes
	s.nbytes = 0

	s.mutex.Unlock()
	return
}

func (s *SPI) Write(p []byte) (n int, err error) {
	s.mutex.Lock()

	// Set up the transaction
	s.txBuffer = p
	s.state = stateWriting

	// Enable the DRE interrupt that will write the bytes from the buffer
	chip.SERCOM_SPIM[s.SERCOM].INTENSET.SetDRE(true)

	s.busMutex.Lock()

	// ...
	// Wait for transactions to complete
	// ...

	// Disable the DRE interrupt since the TX buffer will be unset
	chip.SERCOM_SPIM[s.SERCOM].INTENSET.SetDRE(false)

	// Reset the state
	s.state = stateIdle
	s.txBuffer = nil

	n = s.nbytes
	s.nbytes = 0

	s.mutex.Unlock()
	return
}

func (s *SPI) Transact(rx []byte, tx []byte) error {
	// The length of the buffers must match
	if len(rx) != len(tx) {
		// TODO: Wrap this error adding a more specific error message
		return peripheral.ErrInvalidBuffer
	}

	s.mutex.Lock()

	// Set up the transaction
	s.rxBuffer = rx
	s.txBuffer = tx
	s.state = stateTransacting

	// Enable both the RXC and DRE buffer so that bytes can be transmitted and received at the same time
	chip.SERCOM_SPIM[s.SERCOM].INTENSET.SetDRE(true)
	chip.SERCOM_SPIM[s.SERCOM].INTENSET.SetRXC(true)

	s.busMutex.Lock()

	// ...
	// Wait for transactions to complete
	// ...

	// Disable both the RXC and DRE interrupts
	chip.SERCOM_SPIM[s.SERCOM].INTENSET.SetDRE(false)
	chip.SERCOM_SPIM[s.SERCOM].INTENSET.SetRXC(true)

	// Reset the state
	s.state = stateIdle
	s.txBuffer = nil
	s.rxBuffer = nil
	s.nbytes = 0

	s.mutex.Unlock()
	return nil
}

func (s *SPI) Select() {
	if !s.hardwareCS {
		s.cs.Set(false)
	}
}

func (s *SPI) Deselect() {
	if !s.hardwareCS {
		s.cs.Set(true)
	}
}

func irqHandler() {
	sercom := int(chip.SystemControl.ICSR.GetVECTACTIVE()-16) - int(samd21.IRQ_SERCOM0)
	switch {
	case chip.SERCOM_SPIM[sercom].INTFLAG.GetRXC():
		rxcHandler(sercom)
	case chip.SERCOM_SPIM[sercom].INTFLAG.GetDRE():
		dreHandler(sercom)
	}
}

func rxcHandler(sercom int) {
	s := spi[sercom]

	if s.state == stateDone {
		return
	}

	if s.nbytes < len(s.rxBuffer) {
		// Receive the incoming byte
		b := byte(chip.SERCOM_SPIM[sercom].DATA.GetDATA())
		s.rxBuffer[s.nbytes] = b
	}

	if s.state == stateReading {
		// NOTE: This increment is placed inside of this if-statement block to prevent interfering with transactions
		//       where transmitting is driving the state.
		s.nbytes++

		if s.nbytes < len(s.rxBuffer) {
			// Write another byte to the data register in order to read the next byte
			chip.SERCOM_SPIM[sercom].DATA.SetDATA(0)
		} else {
			// Release the bus lock
			chip.SERCOM_SPIM[sercom].INTENCLR.SetRXC(true)

			s.state = stateDone
			s.busMutex.Unlock()
		}
	} else if s.state == stateTransacting {
		if s.nbytes >= len(s.txBuffer) {
			chip.SERCOM_SPIM[sercom].INTENCLR.SetRXC(true)

			s.state = stateDone
			s.busMutex.Unlock()
		}
	}
}

func dreHandler(sercom int) {
	s := spi[sercom]

	if s.state == stateDone {
		return
	}

	if s.nbytes < len(s.txBuffer) {
		// Transmit the outgoing byte
		b := s.txBuffer[s.nbytes]
		chip.SERCOM_SPIM[sercom].DATA.SetDATA(uint16(b))
		s.nbytes++
	}

	if s.nbytes >= len(s.txBuffer) {
		// Disable this interrupt
		chip.SERCOM_SPIM[sercom].INTENCLR.SetDRE(true)

		if s.state == stateWriting {
			s.state = stateDone
			s.busMutex.Unlock()
		}
	}
	// NOTE: The transaction is complete when the last byte is received when the state is stateTransacting
}
