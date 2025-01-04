//go:build atsamd21 && !generic

package spi

import (
	"sync"

	"peripheral"
	"peripheral/pin"

	"runtime/arm/cortexm/sam/atsamd21"
	"runtime/arm/cortexm/sam/atsamd21/support/sercom/spim"
	"runtime/arm/cortexm/support/systemcontrol"
)

var (
	SPI0 = &_spi{SERCOM: 0}
	SPI1 = &_spi{SERCOM: 1}
	SPI2 = &_spi{SERCOM: 2}
	SPI3 = &_spi{SERCOM: 3}
	SPI4 = &_spi{SERCOM: 4}
	SPI5 = &_spi{SERCOM: 5}
	spi  = [6]*_spi{
		SPI0,
		SPI1,
		SPI2,
		SPI3,
		SPI4,
		SPI5,
	}
)

const (
	HostMode   = spim.CtrlaModeSpiMaster
	ClientMode = spim.CtrlaModeSpiSlave

	MSB = spim.CtrlaDordMsb
	LSB = spim.CtrlaDordLsb

	Frame            = spim.CtrlaFormSpiFrame
	FrameWithAddress = spim.CtrlaFormSpiFrameWithAddr

	LeadingEdge  = spim.CtrlaCphaLeadingEdge
	TrailingEdge = spim.CtrlaCphaTrailingEdge

	IdleLow  = spim.CtrlaCpolIdleLow
	IdleHigh = spim.CtrlaCpolIdleHigh

	errStrInvalidPin = "(spi): invalid pin configuration"
)

const (
	stateIdle uint8 = iota
	stateReading
	stateWriting
	stateTransacting
	stateDone
)

type _spi struct {
	atsamd21.SERCOM
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
	DataOrder      spim.ConstantSercomSpimCtrlaDordType
	Form           spim.ConstantSercomSpimCtrlaFormType
	HardwareSelect bool
	Mode           spim.ConstantSercomSpimCtrlaModeType
	Phase          spim.ConstantSercomSpimCtrlaCphaType
	Polarity       spim.ConstantSercomSpimCtrlaCpolType
	ReceiveEnabled bool
}

func spiValidateDIPO(do, sck, cs int, mssen bool) (dopo spim.ConstantSercomSpimCtrlaDopoType, ok bool) {
	ok = true
	if do == 0 {
		if sck == 1 {
			dopo = spim.CtrlaDopoPad0
		} else if sck != 3 {
			dopo = spim.CtrlaDopoPad3
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
			dopo = spim.CtrlaDopoPad1
		} else {
			return 0, false
		}

		if mssen && cs != 1 {
			return 0, false
		}
	} else if do == 3 {
		if sck == 1 {
			dopo = spim.CtrlaDopoPad2
		} else {
			return 0, false
		}

		if mssen && cs != 2 {
			return 0, false
		}
	}
	return
}

func (s *_spi) Configure(config Config) {
	var mode pin.PMUXFunction
	var doPad spim.ConstantSercomSpimCtrlaDopoType
	var diPad spim.ConstantSercomSpimCtrlaDipoType
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
			diPad = spim.CtrlaDipoPad0
		case 1:
			diPad = spim.CtrlaDipoPad1
		case 2:
			diPad = spim.CtrlaDipoPad2
		case 3:
			diPad = spim.CtrlaDipoPad3
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
	spim.Spim[s.SERCOM].Ctrla.SetSwrst(true)
	s.Synchronize()

	// Enable the SERCOM
	s.SetPMEnabled(true)

	// Calculate the BAUD value
	baud := s.Baud(config.BaudHz)

	// Set up the registers
	spim.Spim[s.SERCOM].Baud.SetBaud(baud)

	spim.Spim[s.SERCOM].Ctrla.SetDord(config.DataOrder)
	spim.Spim[s.SERCOM].Ctrla.SetForm(config.Form)
	spim.Spim[s.SERCOM].Ctrla.SetMode(config.Mode)
	spim.Spim[s.SERCOM].Ctrla.SetCpha(config.Phase)
	spim.Spim[s.SERCOM].Ctrla.SetCpol(config.Polarity)
	spim.Spim[s.SERCOM].Ctrla.SetDipo(diPad)
	spim.Spim[s.SERCOM].Ctrla.SetDopo(doPad)
	spim.Spim[s.SERCOM].Ctrla.SetCpol(config.Polarity)

	spim.Spim[s.SERCOM].Ctrlb.SetRxen(config.ReceiveEnabled)
	spim.Spim[s.SERCOM].Ctrlb.SetMssen(config.HardwareSelect)
	switch config.CharacterSize {
	case 9:
		spim.Spim[s.SERCOM].Ctrlb.SetChsize(spim.CtrlbChsize9Bit)
	default:
		spim.Spim[s.SERCOM].Ctrlb.SetChsize(spim.CtrlbChsize8Bit)
	}

	// Set the interrupt handler
	s.SetHandler(irqHandler)

	// Enable interrupts
	s.Irq().EnableIRQ()

	// Enable the peripheral
	spim.Spim[s.SERCOM].Ctrla.SetEnable(true)
	for spim.Spim[s.SERCOM].Syncbusy.GetEnable() {
	}

	s.cs = config.CS
	s.hardwareCS = config.HardwareSelect

	// Lock the bus mutex
	s.busMutex.Lock()
}

func (s *_spi) Read(p []byte) (n int, err error) {
	s.mutex.Lock()

	// Set up the transaction
	s.rxBuffer = p
	s.state = stateReading

	// Enable receive interrupt so the incoming data can be written to the RX buffer
	spim.Spim[s.SERCOM].Intenset.SetRxc(true)

	// Write a byte to the DATA register to begin the transactions
	spim.Spim[s.SERCOM].Data.SetData(0)

	s.busMutex.Lock()

	// ...
	// Wait for transactions to complete
	// ...

	// Disable the receive interrupt since the RX buffer will be unset
	spim.Spim[s.SERCOM].Intenset.SetRxc(false)

	// Reset the state
	s.state = stateIdle
	s.rxBuffer = nil

	n = s.nbytes
	s.nbytes = 0

	s.mutex.Unlock()
	return
}

func (s *_spi) Write(p []byte) (n int, err error) {
	s.mutex.Lock()

	// Set up the transaction
	s.txBuffer = p
	s.state = stateWriting

	// Enable the DRE interrupt that will write the bytes from the buffer
	spim.Spim[s.SERCOM].Intenset.SetDre(true)

	s.busMutex.Lock()

	// ...
	// Wait for transactions to complete
	// ...

	// Disable the DRE interrupt since the TX buffer will be unset
	spim.Spim[s.SERCOM].Intenset.SetDre(false)

	// Reset the state
	s.state = stateIdle
	s.txBuffer = nil

	n = s.nbytes
	s.nbytes = 0

	s.mutex.Unlock()
	return
}

func (s *_spi) Transact(rx []byte, tx []byte) error {
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
	spim.Spim[s.SERCOM].Intenset.SetDre(true)
	spim.Spim[s.SERCOM].Intenset.SetRxc(true)

	s.busMutex.Lock()

	// ...
	// Wait for transactions to complete
	// ...

	// Disable both the RXC and DRE interrupts
	spim.Spim[s.SERCOM].Intenset.SetDre(false)
	spim.Spim[s.SERCOM].Intenset.SetRxc(true)

	// Reset the state
	s.state = stateIdle
	s.txBuffer = nil
	s.rxBuffer = nil
	s.nbytes = 0

	s.mutex.Unlock()
	return nil
}

func (s *_spi) Select() {
	if !s.hardwareCS {
		s.cs.Set(false)
	}
}

func (s *_spi) Deselect() {
	if !s.hardwareCS {
		s.cs.Set(true)
	}
}

func irqHandler() {
	sercom := int(systemcontrol.SystemControl.Icsr.GetVectactive()-16) - int(atsamd21.IRQ_SERCOM0)
	switch {
	case spim.Spim[sercom].Intflag.GetRxc():
		rxcHandler(sercom)
	case spim.Spim[sercom].Intflag.GetDre():
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
		b := byte(spim.Spim[sercom].Data.GetData())
		s.rxBuffer[s.nbytes] = b
	}

	if s.state == stateReading {
		// NOTE: This increment is placed inside of this if-statement block to prevent interfering with transactions
		//       where transmitting is driving the state.
		s.nbytes++

		if s.nbytes < len(s.rxBuffer) {
			// Write another byte to the data register in order to read the next byte
			spim.Spim[sercom].Data.SetData(0)
		} else {
			// Release the bus lock
			spim.Spim[sercom].Intenclr.SetRxc(true)

			s.state = stateDone
			s.busMutex.Unlock()
		}
	} else if s.state == stateTransacting {
		if s.nbytes >= len(s.txBuffer) {
			spim.Spim[sercom].Intenclr.SetRxc(true)

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
		spim.Spim[sercom].Data.SetData(uint16(b))
		s.nbytes++
	}

	if s.nbytes >= len(s.txBuffer) {
		// Disable this interrupt
		spim.Spim[sercom].Intenclr.SetDre(true)

		if s.state == stateWriting {
			s.state = stateDone
			s.busMutex.Unlock()
		}
	}
	// NOTE: The transaction is complete when the last byte is received when the state is stateTransacting
}
