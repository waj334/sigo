//go:build atsamd21 && !generic

package i2c

import (
	"errors"
	"peripheral"
	"peripheral/pin"
	"runtime/arm/cortexm/sam/atsamd21"
	"runtime/arm/cortexm/sam/atsamd21/support/sercom/i2cm"
	"runtime/arm/cortexm/support/systemcontrol"
	"sync"
)

const (
	StandardAndFastMode = i2cm.CtrlaSpeedStandardAndFastMode
	FastMode            = i2cm.CtrlaSpeedFastplusMode
	HighSpeedMode       = i2cm.CtrlaSpeedHighSpeedMode

	NoSDAHold    = i2cm.CtrlaSdaholdDisable
	SDAHold75NS  = i2cm.CtrlaSdahold75Ns
	SDAHold450NS = i2cm.CtrlaSdahold450Ns
	SDAHold600NS = i2cm.CtrlaSdahold600Ns

	NoTimeout    = i2cm.CtrlaInactoutDisable
	Timeout55US  = i2cm.CtrlaInactout55Us
	Timeout105US = i2cm.CtrlaInactout105Us
	Timeout205US = i2cm.CtrlaInactout205Us

	busStateUnknown = 0
	busStateIdle    = 1
	busStateOwner   = 2
	busStateBusy    = 3

	cmdStart = 0x1
	cmdRead  = 0x2
	cmdStop  = 0x3
)

const (
	stateIdle = iota
	stateSendMasterCode
	stateSendAddress
	stateWriting
	stateReading
	stateDone
	stateError
)

type MasterCode uint8

const (
	MasterCode1 MasterCode = iota + 0x09
	MasterCode2
	MasterCode3
	MasterCode4
	MasterCode5
	MasterCode6
	MasterCode7
)

var (
	I2C0 = &_i2c{SERCOM: 0}
	I2C1 = &_i2c{SERCOM: 1}
	I2C2 = &_i2c{SERCOM: 2}
	I2C3 = &_i2c{SERCOM: 3}
	I2C4 = &_i2c{SERCOM: 4}
	I2C5 = &_i2c{SERCOM: 5}
	i2c  = [6]*_i2c{
		I2C0,
		I2C1,
		I2C2,
		I2C3,
		I2C4,
		I2C5,
	}

	errLENERR   = errors.New("i2c: transaction length error")
	errSEXTTOUT = errors.New("i2c: client SCL low extend time-out")
	errMEXTTOUT = errors.New("i2c: host SCL low extend time-out")
	errLOWTOUT  = errors.New("i2c: SCL low time-out")
	errARBLOST  = errors.New("i2c: arbitration lost")
	errBUSERR   = errors.New("i2c: bus error")
	errRXNACK   = errors.New("i2c: received not acknowledge")
)

type _i2c struct {
	atsamd21.SERCOM
	address    uint16
	mutex      sync.Mutex
	busMutex   sync.Mutex
	masterCode MasterCode

	buf    []byte
	nbytes int
	state  int
	fSCL   uint32
}

type Config struct {
	SDA     pin.Pin
	SCL     pin.Pin
	SDA_OUT pin.Pin
	SCL_OUT pin.Pin

	ClockSpeedHz uint32
	MasterCode   MasterCode

	Speed      i2cm.ConstantSercomI2cmCtrlaSpeedType
	SDAHold    i2cm.ConstantSercomI2cmCtrlaSdaholdType
	BusTimeout i2cm.ConstantSercomI2cmCtrlaInactoutType

	SCLStretch                bool
	SCLLowTimeout             bool
	ExtendClientSCLLowTimeout bool
	ExtendHostSCLLowTimeout   bool

	FourWireOperation bool
	RunInStandby      bool
}

func (c *Config) validate() bool {
	if c.SDA.GetPAD() != 0 && c.SDA.GetAltPAD() != 0 {
		return false
	}

	if c.SCL.GetPAD() != 1 && c.SCL.GetAltPAD() != 1 {
		return false
	}

	if c.SDA_OUT != pin.NoPin && c.SDA_OUT.GetPAD() != 2 && c.SDA_OUT.GetAltPAD() != 2 {
		return false
	}

	if c.SCL_OUT != pin.NoPin && c.SCL_OUT.GetPAD() != 3 && c.SCL_OUT.GetAltPAD() != 3 {
		return false
	}

	return true
}

func (i *_i2c) Configure(config Config) error {
	// Validate pin configuration
	if !config.validate() {
		return peripheral.ErrInvalidPinout
	}

	// Calculate the baud
	var baudLow, baudHigh uint32
	baudLow, baudHigh, ok := CalculateBaudValue(atsamd21.SercomRefFrequency, config.ClockSpeedHz)
	if !ok {
		return peripheral.ErrInvalidConfig
	}

	// Set pin configurations
	if config.SDA.GetPAD() == 0 {
		config.SDA.SetPMUX(pin.PMUXFunctionC, true)
	} else {
		config.SDA.SetPMUX(pin.PMUXFunctionD, true)
	}

	if config.SCL.GetPAD() == 1 {
		config.SCL.SetPMUX(pin.PMUXFunctionC, true)
	} else {
		config.SCL.SetPMUX(pin.PMUXFunctionD, true)
	}

	if config.SDA_OUT != pin.NoPin {
		if config.SDA_OUT.GetPAD() == 2 {
			config.SDA_OUT.SetPMUX(pin.PMUXFunctionC, true)
		} else {
			config.SDA_OUT.SetPMUX(pin.PMUXFunctionD, true)
		}
	}

	if config.SCL_OUT != pin.NoPin {
		if config.SCL_OUT.GetPAD() == 3 {
			config.SCL_OUT.SetPMUX(pin.PMUXFunctionC, true)
		} else {
			config.SCL_OUT.SetPMUX(pin.PMUXFunctionD, true)
		}
	}

	// Set the master code
	if config.MasterCode == 0 {
		// Default 1b00001001
		config.MasterCode = MasterCode1
	}

	// First, reset the SERCOM
	i2cm.I2cm[i.SERCOM].Ctrla.SetSwrst(true)
	i.Synchronize()

	// Enable the SERCOM in PM
	i.SERCOM.SetPMEnabled(true)

	// Set the SERCOM mode before writing to CTRLA and CTRLB
	i2cm.I2cm[i.SERCOM].Ctrla.SetMode(i2cm.CtrlaModeI2cMaster)

	// Enable smart mode
	i2cm.I2cm[i.SERCOM].Ctrlb.SetSmen(true)
	i.Synchronize()

	// Set options
	i2cm.I2cm[i.SERCOM].Ctrla.SetSdahold(config.SDAHold)
	i2cm.I2cm[i.SERCOM].Ctrla.SetSpeed(config.Speed)
	i2cm.I2cm[i.SERCOM].Ctrla.SetPinout(config.FourWireOperation)
	i2cm.I2cm[i.SERCOM].Ctrla.SetRunstdby(config.RunInStandby)
	i2cm.I2cm[i.SERCOM].Ctrla.SetLowtouten(config.SCLLowTimeout)
	i2cm.I2cm[i.SERCOM].Ctrla.SetMexttoen(config.ExtendHostSCLLowTimeout)
	i2cm.I2cm[i.SERCOM].Ctrla.SetSexttoen(config.ExtendClientSCLLowTimeout)
	i2cm.I2cm[i.SERCOM].Ctrla.SetInactout(config.BusTimeout)

	if config.Speed == HighSpeedMode {
		// _i2c High-speed (Hs) mode requires CTRLA.SCLSM=1.
		i2cm.I2cm[i.SERCOM].Ctrla.SetSclsm(true)

		// Set the speed during HS mode
		i2cm.I2cm[i.SERCOM].Baud.SetHsbaud(uint8(baudHigh))
		i2cm.I2cm[i.SERCOM].Baud.SetHsbaudlow(uint8(baudLow))

		// Configure Fast-Mode for 400 KHz
		baudLow, baudHigh, ok = CalculateBaudValue(atsamd21.SercomRefFrequency, 400_000)
		if !ok {
			return peripheral.ErrInvalidConfig
		}
	} else {
		i2cm.I2cm[i.SERCOM].Ctrla.SetSclsm(config.SCLStretch)
	}

	// Configure baud rate for Fast-Mode
	// NOTE: Fast-Mode is still used in HS Mode to transmit Master Code and the Address bits
	i2cm.I2cm[i.SERCOM].Baud.SetBaud(uint8(baudHigh))
	i2cm.I2cm[i.SERCOM].Baud.SetBaudlow(uint8(baudLow))

	// Enable the SERCOM
	i2cm.I2cm[i.SERCOM].Ctrla.SetEnable(true)
	i.Synchronize()

	// Set initial bus state to idle
	i2cm.I2cm[i.SERCOM].Status.SetBusstate(busStateIdle)
	i.Synchronize()

	// Enable interrupts
	i.Irq().EnableIRQ()
	i.SetHandler(irqHandler)
	i2cm.I2cm[i.SERCOM].Intenset.SetError(true)
	i2cm.I2cm[i.SERCOM].Intenset.SetSb(true)
	i2cm.I2cm[i.SERCOM].Intenset.SetMb(true)

	// Lock the bus mutex
	i.busMutex.Lock()

	return nil
}

func (i *_i2c) Read(b []byte) (n int, err error) {
	i.mutex.Lock()
	n, err = i.read(b)
	i.mutex.Unlock()
	return
}

func (i *_i2c) Write(b []byte) (n int, err error) {
	i.mutex.Lock()
	n, err = i.write(b)
	i.mutex.Unlock()
	return
}

func (i *_i2c) SetAddress(addr uint16) {
	i.address = addr
}

func (i *_i2c) SetClockFrequency(clockSpeedHz uint32) bool {
	var baudLow, baudHigh uint32
	baudLow, baudHigh, ok := CalculateBaudValue(atsamd21.SercomRefFrequency, clockSpeedHz)
	if !ok {
		return false
	}

	// Disable the SERCOM first
	i2cm.I2cm[i.SERCOM].Ctrla.SetEnable(false)
	i.Synchronize()

	if i2cm.I2cm[i.SERCOM].Ctrla.GetSpeed() == HighSpeedMode {
		// Set the speed during HS mode
		i2cm.I2cm[i.SERCOM].Baud.SetHsbaud(uint8(baudHigh))
		i2cm.I2cm[i.SERCOM].Baud.SetHsbaudlow(uint8(baudLow))

		// Configure Fast-Mode for 400 KHz
		baudLow, baudHigh, ok = CalculateBaudValue(atsamd21.SercomRefFrequency, 400_000)
		if !ok {
			return false
		}
	}

	// Configure baud rate for Fast-Mode
	// NOTE: Fast-Mode is still used in HS Mode to transmit Master Code and the Address bits
	i2cm.I2cm[i.SERCOM].Baud.SetBaud(uint8(baudHigh))
	i2cm.I2cm[i.SERCOM].Baud.SetBaudlow(uint8(baudLow))

	// Enable the SERCOM again
	i2cm.I2cm[i.SERCOM].Ctrla.SetEnable(true)
	i.Synchronize()

	// Reset the bus state
	i2cm.I2cm[i.SERCOM].Status.SetBusstate(busStateIdle)
	i.Synchronize()

	i.fSCL = clockSpeedHz

	return true
}

func (i *_i2c) GetClockFrequency() uint32 {
	return i.fSCL
}

func (i *_i2c) WriteAddress(addr uint16, b []byte) (n int, err error) {
	i.mutex.Lock()

	// First, set the address
	// This call does not set the address register
	i.SetAddress(addr)

	// Perform the transfer
	n, err = i.write(b)

	i.mutex.Unlock()
	return
}

func (i *_i2c) ReadAddress(addr uint16, b []byte) (n int, err error) {
	i.mutex.Lock()

	// First, set the address
	i.SetAddress(addr)

	// Perform the transfer
	n, err = i.read(b)

	i.mutex.Unlock()
	return
}

func (i *_i2c) initTransfer(b []byte, write bool) {
	isHighSpeed := false
	if i2cm.I2cm[i.SERCOM].Ctrla.GetSpeed() == HighSpeedMode {
		i.state = stateSendMasterCode
		isHighSpeed = true
	} else {
		i.state = stateSendAddress
	}

	if i.state == stateSendMasterCode {
		// Transmit the master code
		var ADDR i2cm.RegisterAddrType
		ADDR.SetHs(false)
		ADDR.SetAddr(uint16(i.masterCode))
		i2cm.I2cm[i.SERCOM].Addr = ADDR
		i.Synchronize()
		i.state = stateSendAddress
	}

	if i.state == stateSendAddress {
		// Set the buffer slice that will receive data or be used to transmit data
		i.buf = b

		// Shift the address left by 1 to make room for the direction bit
		address := i.address << 1
		if !write {
			address |= 1 // Set read bit
			i.state = stateReading
		} else {
			// NOTE: Write bit is zero, so just the shift is enough
			i.state = stateWriting
		}

		// Transmit the address
		var ADDR i2cm.RegisterAddrType
		ADDR.SetHs(isHighSpeed)
		ADDR.SetAddr(address)
		i2cm.I2cm[i.SERCOM].Addr = ADDR
		i.Synchronize()
	}
}

func (i *_i2c) read(b []byte) (n int, err error) {
	// Transmit the master code and/or address
	i.initTransfer(b, false)

	// Wait for the transfer to complete
	i.busMutex.Lock()

	// Async transfer should take place now
	// interrupt will unlock busMutex when transfer is complete

	// Results
	n = i.nbytes
	if i.state == stateError {
		if i.state == stateError {
			err = i.statusError()
		}
	}

	// Reset
	i.resetState()

	return
}

func (i *_i2c) write(b []byte) (n int, err error) {
	// Transmit the master code and/or address
	i.initTransfer(b, true)

	// Wait for the transfer to complete
	i.busMutex.Lock()

	// Async transfer should take place now
	// interrupt will unlock busMutex when transfer is complete

	// Results
	n = i.nbytes
	if i.state == stateError {
		err = i.statusError()
	}

	// Reset
	i.resetState()

	return
}

func (i *_i2c) resetState() {
	i.nbytes = 0
	i.state = stateIdle
	i.buf = nil
}

func (i *_i2c) syncSysop() {
	for i2cm.I2cm[i.SERCOM].Syncbusy.GetSysop() {
		// Wait for sysop to clear
	}
}

func CalculateBaudValue(srcClkFreq uint32, i2cClkSpeed uint32) (uint32, uint32, bool) {
	var baudValue, baudLow, baudHigh uint32
	fSrcClkFreq := float32(srcClkFreq)
	fI2cClkSpeed := float32(i2cClkSpeed)
	var fBaudValue float32

	// Reference clock frequency must be at least two times the baud rate
	if srcClkFreq < (2 * i2cClkSpeed) {
		return 0, 0, false
	}

	if i2cClkSpeed > 1_000_000 {
		// HS mode baud calculation
		fBaudValue = (fSrcClkFreq / fI2cClkSpeed) - 2.0
		baudValue = uint32(fBaudValue)
		baudLow = baudValue
		baudHigh = baudValue
	} else {
		// Standard, FM and FM+ baud calculation
		fBaudValue = (fSrcClkFreq / fI2cClkSpeed) - ((fSrcClkFreq * (100.0 / 1_000_000_000.0)) + 10.0)
		baudValue = uint32(fBaudValue)
	}

	if i2cClkSpeed <= 400_000 {
		// For _i2c clock speed up to 400 kHz, the value of BAUD<7:0> determines both SCL_L and SCL_H with SCL_L = SCL_H
		if baudValue > (0xFF * 2) {
			// Set baud rate to the minimum possible value
			baudValue = 0xFF
		} else if baudValue <= 1 {
			// Baud value cannot be 0. Set baud rate to maximum possible value
			baudValue = 1
		} else {
			baudValue /= 2
		}
		baudLow = baudValue
		baudHigh = baudValue
	} else {
		// To maintain the ratio of SCL_L:SCL_H to 2:1, the max value of BAUD_LOW<15:8>:BAUD<7:0> can be 0xFF:0x7F. Hence BAUD_LOW + BAUD can not exceed 255+127 = 382
		if baudValue >= 382 {
			// Set baud rate to the minimum possible value while maintaining SCL_L:SCL_H to 2:1
			baudValue = (0xFF << 8) | 0x7F
		} else if baudValue <= 3 {
			// Baud value cannot be 0. Set baud rate to maximum possible value while maintaining SCL_L:SCL_H to 2:1
			baudValue = (2 << 8) | 1
		} else {
			// For Fm+ mode, _i2c SCL_L:SCL_H to 2:1
			baudLow = ((baudValue * 2) / 3) << 8
			baudHigh = baudValue / 3
		}
	}
	return baudLow, baudHigh, true
}

func (i *_i2c) statusError() (err error) {
	// Return the respective error
	switch {
	case i2cm.I2cm[i.SERCOM].Status.GetLenerr():
		err = errLENERR
	case i2cm.I2cm[i.SERCOM].Status.GetSexttout():
		err = errSEXTTOUT
	case i2cm.I2cm[i.SERCOM].Status.GetMexttout():
		err = errMEXTTOUT
	case i2cm.I2cm[i.SERCOM].Status.GetLowtout():
		err = errLOWTOUT
	case i2cm.I2cm[i.SERCOM].Status.GetArblost():
		err = errARBLOST
	case i2cm.I2cm[i.SERCOM].Status.GetBuserr():
		err = errBUSERR
	case i2cm.I2cm[i.SERCOM].Status.GetRxnack():
		err = errRXNACK
	default:
		err = nil
	}
	return
}

func irqHandler() {
	sercom := int(systemcontrol.SystemControl.Icsr.GetVectactive()-16) - int(atsamd21.IRQ_SERCOM0)
	i := i2c[sercom]
	switch {
	case i2cm.I2cm[sercom].Intflag.GetError():
		// Error occurred. Allow the read/write operation to resume
		i.state = stateError

		// Clear the ERROR flag
		i2cm.I2cm[sercom].Intflag.SetError(true)

		i.busMutex.Unlock()
	case i2cm.I2cm[sercom].Intflag.GetSb():
		// Client sent data
		// Host sent data
		switch {
		case i2cm.I2cm[sercom].Status.GetRxnack():
			// Send the STOP condition
			i2cm.I2cm[sercom].Ctrlb.SetCmd(cmdStop)
			fallthrough
		case i2cm.I2cm[sercom].Status.GetArblost():
			// There was an issue with the last packet
			i.state = stateError
			i.busMutex.Unlock()
		default:
			if i.state == stateReading {
				// Transmit the next byte
				if len(i.buf) > 0 && i.nbytes < len(i.buf) {
					// NOTE: Smart mode is enabled, so the ACK will be transmitted automatically
					i.buf[i.nbytes] = i2cm.I2cm[sercom].Data.GetData()
					i.nbytes++

					if i.nbytes == len(i.buf) {
						// Send NACK for the next byte
						i2cm.I2cm[sercom].Ctrlb.SetAckact(true)
					} else {
						// Send ACK for the next byte
						i2cm.I2cm[sercom].Ctrlb.SetAckact(false)
					}
					i.Synchronize()
				} else {
					i.state = stateDone

					// Send the STOP condition
					i2cm.I2cm[sercom].Ctrlb.SetCmd(cmdStop)

					// Allow the read/write function to continue
					i.busMutex.Unlock()
				}
			}
		}

		// Clear the SB flag
		i2cm.I2cm[sercom].Intflag.SetSb(true)

	case i2cm.I2cm[sercom].Intflag.GetMb():
		// Host sent data
		switch {
		case i2cm.I2cm[sercom].Status.GetRxnack():
			// Send the STOP condition
			i2cm.I2cm[sercom].Ctrlb.SetCmd(cmdStop)
			fallthrough
		case i2cm.I2cm[sercom].Status.GetArblost():
			// There was an issue with the last packet
			i.state = stateError
			i.busMutex.Unlock()
		default:
			if i.state == stateWriting {
				// Transmit the next byte
				if len(i.buf) > 0 && i.nbytes < len(i.buf) {
					i2cm.I2cm[sercom].Data.SetData(i.buf[i.nbytes])
					i.nbytes++
					i.Synchronize()
				} else {
					i.state = stateDone

					// Send the STOP condition
					i2cm.I2cm[sercom].Ctrlb.SetCmd(cmdStop)

					// Allow the read/write function to continue
					i.busMutex.Unlock()
				}
			}
		}

		// Clear the MB flag
		i2cm.I2cm[sercom].Intflag.SetMB(true)
	}
}
