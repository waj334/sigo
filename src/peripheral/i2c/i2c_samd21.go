//go:build samd21 && !generic

package i2c

import (
	"errors"
	"peripheral"
	"peripheral/pin"
	"runtime/arm/cortexm/sam/chip"
	"runtime/arm/cortexm/sam/samd21"
	"sync"
)

const (
	StandardAndFastMode = chip.SERCOM_I2CM_CTRLA_REG_SPEED_STANDARD_AND_FAST_MODE
	FastMode            = chip.SERCOM_I2CM_CTRLA_REG_SPEED_FASTPLUS_MODE
	HighSpeedMode       = chip.SERCOM_I2CM_CTRLA_REG_SPEED_HIGH_SPEED_MODE

	NoSDAHold    = chip.SERCOM_I2CM_CTRLA_REG_SDAHOLD_DISABLE
	SDAHold75NS  = chip.SERCOM_I2CM_CTRLA_REG_SDAHOLD_75NS
	SDAHold450NS = chip.SERCOM_I2CM_CTRLA_REG_SDAHOLD_450NS
	SDAHold600NS = chip.SERCOM_I2CM_CTRLA_REG_SDAHOLD_600NS

	NoTimeout    = chip.SERCOM_I2CM_CTRLA_REG_INACTOUT_DISABLE
	Timeout55US  = chip.SERCOM_I2CM_CTRLA_REG_INACTOUT_55US
	Timeout105US = chip.SERCOM_I2CM_CTRLA_REG_INACTOUT_105US
	Timeout205US = chip.SERCOM_I2CM_CTRLA_REG_INACTOUT_205US

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
	I2C0 = &I2C{SERCOM: 0}
	I2C1 = &I2C{SERCOM: 1}
	I2C2 = &I2C{SERCOM: 2}
	I2C3 = &I2C{SERCOM: 3}
	I2C4 = &I2C{SERCOM: 4}
	I2C5 = &I2C{SERCOM: 5}
	i2c  = [6]*I2C{
		I2C0,
		I2C1,
		I2C2,
		I2C3,
		I2C4,
		I2C5,
	}

	errLENERR   = errors.New("I2C: transaction length error")
	errSEXTTOUT = errors.New("I2C: client SCL low extend time-out")
	errMEXTTOUT = errors.New("I2C: host SCL low extend time-out")
	errLOWTOUT  = errors.New("I2C: SCL low time-out")
	errARBLOST  = errors.New("I2C: arbitration lost")
	errBUSERR   = errors.New("I2C: bus error")
	errRXNACK   = errors.New("I2C: received not acknowledge")
)

type I2C struct {
	samd21.SERCOM
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

	Speed      chip.SERCOM_I2CM_CTRLA_REG_SPEED
	SDAHold    chip.SERCOM_I2CM_CTRLA_REG_SDAHOLD
	BusTimeout chip.SERCOM_I2CM_CTRLA_REG_INACTOUT

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

func (i *I2C) Configure(config Config) error {
	// Validate pin configuration
	if !config.validate() {
		return peripheral.ErrInvalidPinout
	}

	// Calculate the baud
	var baudLow, baudHigh uint32
	baudLow, baudHigh, ok := CalculateBaudValue(samd21.SERCOM_REF_FREQUENCY, config.ClockSpeedHz)
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
	chip.SERCOM_I2CM[i.SERCOM].CTRLA.SetSWRST(true)
	i.Synchronize()

	// Enable the SERCOM in PM
	i.SERCOM.SetPMEnabled(true)

	// Set the SERCOM mode before writing to CTRLA and CTRLB
	chip.SERCOM_I2CM[i.SERCOM].CTRLA.SetMODE(chip.SERCOM_I2CM_CTRLA_REG_MODE_I2C_MASTER)

	// Enable smart mode
	chip.SERCOM_I2CM[i.SERCOM].CTRLB.SetSMEN(true)
	i.Synchronize()

	// Set options
	chip.SERCOM_I2CM[i.SERCOM].CTRLA.SetSDAHOLD(config.SDAHold)
	chip.SERCOM_I2CM[i.SERCOM].CTRLA.SetSPEED(config.Speed)
	chip.SERCOM_I2CM[i.SERCOM].CTRLA.SetPINOUT(config.FourWireOperation)
	chip.SERCOM_I2CM[i.SERCOM].CTRLA.SetRUNSTDBY(config.RunInStandby)
	chip.SERCOM_I2CM[i.SERCOM].CTRLA.SetLOWTOUTEN(config.SCLLowTimeout)
	chip.SERCOM_I2CM[i.SERCOM].CTRLA.SetMEXTTOEN(config.ExtendHostSCLLowTimeout)
	chip.SERCOM_I2CM[i.SERCOM].CTRLA.SetSEXTTOEN(config.ExtendClientSCLLowTimeout)
	chip.SERCOM_I2CM[i.SERCOM].CTRLA.SetINACTOUT(config.BusTimeout)

	if config.Speed == HighSpeedMode {
		// I2C High-speed (Hs) mode requires CTRLA.SCLSM=1.
		chip.SERCOM_I2CM[i.SERCOM].CTRLA.SetSCLSM(true)

		// Set the speed during HS mode
		chip.SERCOM_I2CM[i.SERCOM].BAUD.SetHSBAUD(uint8(baudHigh))
		chip.SERCOM_I2CM[i.SERCOM].BAUD.SetHSBAUDLOW(uint8(baudLow))

		// Configure Fast-Mode for 400 KHz
		baudLow, baudHigh, ok = CalculateBaudValue(samd21.SERCOM_REF_FREQUENCY, 400_000)
		if !ok {
			return peripheral.ErrInvalidConfig
		}
	} else {
		chip.SERCOM_I2CM[i.SERCOM].CTRLA.SetSCLSM(config.SCLStretch)
	}

	// Configure baud rate for Fast-Mode
	// NOTE: Fast-Mode is still used in HS Mode to transmit Master Code and the Address bits
	chip.SERCOM_I2CM[i.SERCOM].BAUD.SetBAUD(uint8(baudHigh))
	chip.SERCOM_I2CM[i.SERCOM].BAUD.SetBAUDLOW(uint8(baudLow))

	// Enable the SERCOM
	chip.SERCOM_I2CM[i.SERCOM].CTRLA.SetENABLE(true)
	i.Synchronize()

	// Set initial bus state to idle
	chip.SERCOM_I2CM[i.SERCOM].STATUS.SetBUSSTATE(busStateIdle)
	i.Synchronize()

	// Enable interrupts
	i.Irq().EnableIRQ()
	i.SetHandler(irqHandler)
	chip.SERCOM_I2CM[i.SERCOM].INTENSET.SetERROR(true)
	chip.SERCOM_I2CM[i.SERCOM].INTENSET.SetSB(true)
	chip.SERCOM_I2CM[i.SERCOM].INTENSET.SetMB(true)

	// Lock the bus mutex
	i.busMutex.Lock()

	return nil
}

func (i *I2C) Read(b []byte) (n int, err error) {
	i.mutex.Lock()
	n, err = i.read(b)
	i.mutex.Unlock()
	return
}

func (i *I2C) Write(b []byte) (n int, err error) {
	i.mutex.Lock()
	n, err = i.write(b)
	i.mutex.Unlock()
	return
}

func (i *I2C) SetAddress(addr uint16) {
	i.address = addr
}

func (i *I2C) SetClockFrequency(clockSpeedHz uint32) bool {
	var baudLow, baudHigh uint32
	baudLow, baudHigh, ok := CalculateBaudValue(samd21.SERCOM_REF_FREQUENCY, clockSpeedHz)
	if !ok {
		return false
	}

	// Disable the SERCOM first
	chip.SERCOM_I2CM[i.SERCOM].CTRLA.SetENABLE(false)
	i.Synchronize()

	if chip.SERCOM_I2CM[i.SERCOM].CTRLA.GetSPEED() == HighSpeedMode {
		// Set the speed during HS mode
		chip.SERCOM_I2CM[i.SERCOM].BAUD.SetHSBAUD(uint8(baudHigh))
		chip.SERCOM_I2CM[i.SERCOM].BAUD.SetHSBAUDLOW(uint8(baudLow))

		// Configure Fast-Mode for 400 KHz
		baudLow, baudHigh, ok = CalculateBaudValue(samd21.SERCOM_REF_FREQUENCY, 400_000)
		if !ok {
			return false
		}
	}

	// Configure baud rate for Fast-Mode
	// NOTE: Fast-Mode is still used in HS Mode to transmit Master Code and the Address bits
	chip.SERCOM_I2CM[i.SERCOM].BAUD.SetBAUD(uint8(baudHigh))
	chip.SERCOM_I2CM[i.SERCOM].BAUD.SetBAUDLOW(uint8(baudLow))

	// Enable the SERCOM again
	chip.SERCOM_I2CM[i.SERCOM].CTRLA.SetENABLE(true)
	i.Synchronize()

	// Reset the bus state
	chip.SERCOM_I2CM[i.SERCOM].STATUS.SetBUSSTATE(busStateIdle)
	i.Synchronize()

	i.fSCL = clockSpeedHz

	return true
}

func (i *I2C) GetClockFrequency() uint32 {
	return i.fSCL
}

func (i *I2C) WriteAddress(addr uint16, b []byte) (n int, err error) {
	i.mutex.Lock()

	// First, set the address
	// This call does not set the address register
	i.SetAddress(addr)

	// Perform the transfer
	n, err = i.write(b)

	i.mutex.Unlock()
	return
}

func (i *I2C) ReadAddress(addr uint16, b []byte) (n int, err error) {
	i.mutex.Lock()

	// First, set the address
	i.SetAddress(addr)

	// Perform the transfer
	n, err = i.read(b)

	i.mutex.Unlock()
	return
}

func (i *I2C) initTransfer(b []byte, write bool) {
	isHighSpeed := false
	if chip.SERCOM_I2CM[i.SERCOM].CTRLA.GetSPEED() == HighSpeedMode {
		i.state = stateSendMasterCode
		isHighSpeed = true
	} else {
		i.state = stateSendAddress
	}

	if i.state == stateSendMasterCode {
		// Transmit the master code
		var ADDR chip.SERCOM_I2CM_ADDR_REG
		ADDR.SetHS(false)
		ADDR.SetADDR(uint16(i.masterCode))
		chip.SERCOM_I2CM[i.SERCOM].ADDR = ADDR
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
		var ADDR chip.SERCOM_I2CM_ADDR_REG
		ADDR.SetHS(isHighSpeed)
		ADDR.SetADDR(address)
		chip.SERCOM_I2CM[i.SERCOM].ADDR = ADDR
		i.Synchronize()
	}
}

func (i *I2C) read(b []byte) (n int, err error) {
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

func (i *I2C) write(b []byte) (n int, err error) {
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

func (i *I2C) resetState() {
	i.nbytes = 0
	i.state = stateIdle
	i.buf = nil
}

func (i *I2C) syncSysop() {
	for chip.SERCOM_I2CM[i.SERCOM].SYNCBUSY.GetSYSOP() {
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
		// For I2C clock speed up to 400 kHz, the value of BAUD<7:0> determines both SCL_L and SCL_H with SCL_L = SCL_H
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
			// For Fm+ mode, I2C SCL_L:SCL_H to 2:1
			baudLow = ((baudValue * 2) / 3) << 8
			baudHigh = baudValue / 3
		}
	}
	return baudLow, baudHigh, true
}

func (i *I2C) statusError() (err error) {
	// Return the respective error
	switch {
	case chip.SERCOM_I2CM[i.SERCOM].STATUS.GetLENERR():
		err = errLENERR
	case chip.SERCOM_I2CM[i.SERCOM].STATUS.GetSEXTTOUT():
		err = errSEXTTOUT
	case chip.SERCOM_I2CM[i.SERCOM].STATUS.GetMEXTTOUT():
		err = errMEXTTOUT
	case chip.SERCOM_I2CM[i.SERCOM].STATUS.GetLOWTOUT():
		err = errLOWTOUT
	case chip.SERCOM_I2CM[i.SERCOM].STATUS.GetARBLOST():
		err = errARBLOST
	case chip.SERCOM_I2CM[i.SERCOM].STATUS.GetBUSERR():
		err = errBUSERR
	case chip.SERCOM_I2CM[i.SERCOM].STATUS.GetRXNACK():
		err = errRXNACK
	default:
		err = nil
	}
	return
}

func irqHandler() {
	sercom := int(chip.SystemControl.ICSR.GetVECTACTIVE()-16) - int(samd21.IRQ_SERCOM0)
	i := i2c[sercom]
	switch {
	case chip.SERCOM_I2CM[sercom].INTFLAG.GetERROR():
		// Error occurred. Allow the read/write operation to resume
		i.state = stateError

		// Clear the ERROR flag
		chip.SERCOM_I2CM[sercom].INTFLAG.SetERROR(true)

		i.busMutex.Unlock()
	case chip.SERCOM_I2CM[sercom].INTFLAG.GetSB():
		// Client sent data
		// Host sent data
		switch {
		case chip.SERCOM_I2CM[sercom].STATUS.GetRXNACK():
			// Send the STOP condition
			chip.SERCOM_I2CM[sercom].CTRLB.SetCMD(cmdStop)
			fallthrough
		case chip.SERCOM_I2CM[sercom].STATUS.GetARBLOST():
			// There was an issue with the last packet
			i.state = stateError
			i.busMutex.Unlock()
		default:
			if i.state == stateReading {
				// Transmit the next byte
				if len(i.buf) > 0 && i.nbytes < len(i.buf) {
					// NOTE: Smart mode is enabled, so the ACK will be transmitted automatically
					i.buf[i.nbytes] = chip.SERCOM_I2CM[sercom].DATA.GetDATA()
					i.nbytes++

					if i.nbytes == len(i.buf) {
						// Send NACK for the next byte
						chip.SERCOM_I2CM[sercom].CTRLB.SetACKACT(true)
					} else {
						// Send ACK for the next byte
						chip.SERCOM_I2CM[sercom].CTRLB.SetACKACT(false)
					}
					i.Synchronize()
				} else {
					i.state = stateDone

					// Send the STOP condition
					chip.SERCOM_I2CM[sercom].CTRLB.SetCMD(cmdStop)

					// Allow the read/write function to continue
					i.busMutex.Unlock()
				}
			}
		}

		// Clear the SB flag
		chip.SERCOM_I2CM[sercom].INTFLAG.SetSB(true)

	case chip.SERCOM_I2CM[sercom].INTFLAG.GetMB():
		// Host sent data
		switch {
		case chip.SERCOM_I2CM[sercom].STATUS.GetRXNACK():
			// Send the STOP condition
			chip.SERCOM_I2CM[sercom].CTRLB.SetCMD(cmdStop)
			fallthrough
		case chip.SERCOM_I2CM[sercom].STATUS.GetARBLOST():
			// There was an issue with the last packet
			i.state = stateError
			i.busMutex.Unlock()
		default:
			if i.state == stateWriting {
				// Transmit the next byte
				if len(i.buf) > 0 && i.nbytes < len(i.buf) {
					chip.SERCOM_I2CM[sercom].DATA.SetDATA(i.buf[i.nbytes])
					i.nbytes++
					i.Synchronize()
				} else {
					i.state = stateDone

					// Send the STOP condition
					chip.SERCOM_I2CM[sercom].CTRLB.SetCMD(cmdStop)

					// Allow the read/write function to continue
					i.busMutex.Unlock()
				}
			}
		}

		// Clear the MB flag
		chip.SERCOM_I2CM[sercom].INTFLAG.SetMB(true)
	}
}
