//go:build same51 && !generic

package can

import (
	"peripheral"
	"peripheral/pin"
	"runtime/arm/cortexm/sam/chip"
	"runtime/arm/cortexm/sam/samx51"
	"sync"
	"unsafe"
)

//sigo:extern memset memset
func memset(ptr unsafe.Pointer, value int, num uintptr) unsafe.Pointer

const (
	FIFOMode  = false
	QueueMode = true

	ErrFrameBufferTooSmall CANError = -1
	ErrFIFOOutOfRange      CANError = -2
	ErrQueueFull           CANError = -3

	RejectAll     FilterMode = 0
	AcceptInFIFO0 FilterMode = 1
	AcceptInFIFO1 FilterMode = 2

	FIFO0 FIFO = 0
	FIFO1 FIFO = 1

	Range       FilterType = 0
	Dual        FilterType = 1
	Classic     FilterType = 2
	RangeNoMask FilterType = 3

	Disable                 FilterConfig = 0
	StoreInFIFO0            FilterConfig = 1
	StoreInFIFO1            FilterConfig = 2
	Reject                  FilterConfig = 3
	Priority                FilterConfig = 4
	PriorityAndStoreInFIFO0 FilterConfig = 5
	PriorityAndStoreInFIF1  FilterConfig = 6
	StoreInRXBuffer         FilterConfig = 7
)

var (
	CAN0 = &CAN{
		index: 0,
	}

	CAN1 = &CAN{
		index: 1,
	}

	can = [2]*CAN{
		CAN0,
		CAN1,
	}

	DefaultTiming = BitTiming{
		isDefault: true,
	}
)

type CANError int
type FilterMode uint8
type FIFO uint8
type FilterType uint8
type FilterConfig uint8

func (e CANError) Error() string {
	switch e {
	case 0:
		return "no error"
	case -1:
		return "frame buffer too small"
	case -2:
		return "FIFO index is out of range"
	case -3:
		return "queue is full"
	default:
		return "unknown error"
	}
}

type CAN struct {
	module *chip.CAN_TYPE

	rx0Buffer []uint32
	rx1Buffer []uint32
	txBuffer  []uint32

	rx0Index uint8
	rx1Index uint8
	txIndex  uint8
	index    int8

	rx0DataLength DataLengthCode
	rx1DataLength DataLengthCode
	fifo          FIFO
	mutex         sync.Mutex

	onNewMessage    func(FIFO)
	onMessageLost   func(FIFO)
	onProtocolError func()

	standardFilters []chip.CAN_CAN_SIDFE_0_REG
	extendedFilters []chip.CAN_XIDFE_TYPE
}

type BitTiming struct {
	JumpWidth         uint8
	AfterSample       uint8
	BeforeSample      uint8
	Prescaler         uint16
	DelayCompensation bool

	isDefault bool
}

type Filter struct {
	ID1    uint32
	ID2    uint32
	Type   FilterType
	Config FilterConfig
}

func (f *Filter) toStandardFilter() (result chip.CAN_CAN_SIDFE_0_REG) {
	result.SetSFT(chip.CAN_CAN_SIDFE_0_REG_SFT(f.Type & 0x3))
	result.SetSFEC(chip.CAN_CAN_SIDFE_0_REG_SFEC(f.Config & 0x7))
	result.SetSFID1(uint16(f.ID1 & 0x3FF))
	result.SetSFID2(uint16(f.ID2 & 0x3FF))
	return
}

func (f *Filter) toExtendedFilter() (result chip.CAN_XIDFE_TYPE) {
	result.CAN_XIDFE_0.SetEFEC(chip.CAN_CAN_XIDFE_0_REG_EFEC(f.Config & 0x7))
	result.CAN_XIDFE_0.SetEFID1(f.ID1 & 0xFFFFFFF)
	result.CAN_XIDFE_1.SetEFT(chip.CAN_CAN_XIDFE_1_REG_EFT(f.Type & 0x3))
	result.CAN_XIDFE_1.SetEFID2(f.ID2 & 0xFFFFFFF)
	return
}

type Config struct {
	TX pin.Pin
	RX pin.Pin

	TXQueueMode   bool
	TXNumElements uint8
	TXDataLength  DataLengthCode

	RX0NumElements uint8
	RX0DataLength  DataLengthCode
	RX1NumElements uint8
	RX1DataLength  DataLengthCode

	StandardFilterMode FilterMode
	ExtendedFilterMode FilterMode
	FD                 bool

	StandardFilters []Filter
	ExtendedFilters []Filter

	DataBitTiming    BitTiming
	NominalBitTiming BitTiming

	OnNewMessage    func(FIFO)
	OnMessageLost   func(FIFO)
	OnProtocolError func()
}

func (c *Config) validate() (int8, error) {
	if c.TX == pin.PA22 || c.TX == pin.PA24 {
		if c.RX == pin.PA23 || c.RX == pin.PA25 {
			// CAN 0
			return 0, nil
		}
	} else if c.TX == pin.PB11 || c.TX == pin.PB14 {
		if c.RX == pin.PB13 || c.RX == pin.PB15 {
			// CAN 1
			return 1, nil
		}
	}
	return -1, peripheral.ErrInvalidPinout
}

func (c *CAN) Configure(config Config) error {
	index, err := config.validate()
	if err != nil {
		return err
	} else if index != c.index {
		return peripheral.ErrInvalidConfig
	}

	// Set the pin configurations.
	if index == 0 {
		config.TX.SetPMUX(pin.PMUXFunctionI, true)
		config.RX.SetPMUX(pin.PMUXFunctionI, true)
	} else {
		config.TX.SetPMUX(pin.PMUXFunctionH, true)
		config.RX.SetPMUX(pin.PMUXFunctionH, true)
	}
	c.index = index
	c.module = chip.CAN[c.index]

	// Assert the Initialization (CCCR.INIT) bit so that the CAN module can be configured.
	c.setInit(true)

	// Assert the Configuration Change Enable (CCCR.CCE) bit.
	c.module.CCCR.SetCCE(true)

	// Set up bit timings.
	if !config.NominalBitTiming.isDefault {
		c.module.NBTP.SetNTSEG2(config.NominalBitTiming.AfterSample)
		c.module.NBTP.SetNTSEG1(config.NominalBitTiming.BeforeSample)
		c.module.NBTP.SetNBRP(config.NominalBitTiming.Prescaler)
		c.module.NBTP.SetNSJW(config.NominalBitTiming.JumpWidth)
	}

	if !config.DataBitTiming.isDefault {
		c.module.DBTP.SetDTSEG2(config.DataBitTiming.AfterSample)
		c.module.DBTP.SetDTSEG1(config.DataBitTiming.BeforeSample)
		c.module.DBTP.SetDBRP(uint8(config.DataBitTiming.Prescaler))
		c.module.DBTP.SetDSJW(config.DataBitTiming.JumpWidth)
		c.module.DBTP.SetTDC(config.DataBitTiming.DelayCompensation)
	}

	// Enable/disable CAN FD.
	c.module.CCCR.SetFDOE(config.FD)
	c.module.CCCR.SetBRSE(config.FD)

	//*** Set up transmitter
	if config.TXNumElements > 0 {
		bufLen := FrameLengthInWords(config.TXDataLength) * int(config.TXNumElements)
		c.txBuffer = make([]uint32, bufLen)
		txBufferPtr := uintptr(unsafe.Pointer(unsafe.SliceData(c.txBuffer)))

		c.module.TXESC.SetTBDS(chip.CAN_TXESC_REG_TBDS_DATA8)
		c.module.TXBC.SetTBSA(uint16(txBufferPtr))
		c.module.TXBC.SetNDTB(config.TXNumElements)
		c.module.TXBC.SetTFQM(config.TXQueueMode)
	}
	//***

	//*** Set up receiver.
	if config.RX0NumElements > 0 {
		c.module.RXESC.SetF0DS(translateDLC0(config.RX0DataLength))

		// Set the receive buffer address for FIFO 0.
		bufLen := FrameLengthInWords(config.RX0DataLength) * int(config.RX0NumElements)
		c.rx0Buffer = make([]uint32, bufLen)
		rxBufferPtr := uintptr(unsafe.Pointer(unsafe.SliceData(c.rx0Buffer)))
		c.module.RXF0C.SetF0SA(uint16(rxBufferPtr))
		c.module.RXF0C.SetF0S(config.RX0NumElements)
	}

	if config.RX1NumElements > 0 {
		c.module.RXESC.SetF1DS(translateDLC1(config.RX1DataLength))

		// Set the receive buffer address for FIFO 0.
		bufLen := FrameLengthInWords(config.RX1DataLength) * int(config.RX1NumElements)
		c.rx1Buffer = make([]uint32, bufLen)
		rxBufferPtr := uintptr(unsafe.Pointer(unsafe.SliceData(c.rx1Buffer)))
		c.module.RXF1C.SetF1SA(uint16(rxBufferPtr))
		c.module.RXF1C.SetF1S(config.RX1NumElements)
	}
	//***

	//*** Set up filters.
	c.module.GFC.SetRRFE(false)

	var standardFilterMode chip.CAN_GFC_REG_ANFS
	var extendedFilterMode chip.CAN_GFC_REG_ANFE

	switch config.StandardFilterMode {
	case RejectAll:
		standardFilterMode = chip.CAN_GFC_REG_ANFS_REJECT
	case AcceptInFIFO0:
		standardFilterMode = chip.CAN_GFC_REG_ANFS_RXF0
	case AcceptInFIFO1:
		standardFilterMode = chip.CAN_GFC_REG_ANFS_RXF1
	default:
		return peripheral.ErrInvalidConfig
	}
	c.module.GFC.SetANFS(standardFilterMode)

	switch config.ExtendedFilterMode {
	case RejectAll:
		extendedFilterMode = chip.CAN_GFC_REG_ANFE_REJECT
	case AcceptInFIFO0:
		extendedFilterMode = chip.CAN_GFC_REG_ANFE_RXF0
	case AcceptInFIFO1:
		extendedFilterMode = chip.CAN_GFC_REG_ANFE_RXF1
	default:
		return peripheral.ErrInvalidConfig
	}
	c.module.GFC.SetANFE(extendedFilterMode)

	if numElements := len(config.StandardFilters); numElements > 0 {
		c.standardFilters = make([]chip.CAN_CAN_SIDFE_0_REG, numElements)
		for i := range config.StandardFilters {
			c.standardFilters[i] = config.StandardFilters[i].toStandardFilter()
		}

		flssaPtr := uintptr(unsafe.Pointer(unsafe.SliceData(c.standardFilters)))
		c.module.SIDFC.SetFLSSA(uint16(flssaPtr))
		c.module.SIDFC.SetLSS(uint8(numElements))
	}

	if numElements := len(config.StandardFilters); numElements > 0 {
		c.extendedFilters = make([]chip.CAN_XIDFE_TYPE, numElements)
		for i := range config.ExtendedFilters {
			c.extendedFilters[i] = config.ExtendedFilters[i].toExtendedFilter()
		}

		flesaPtr := uintptr(unsafe.Pointer(unsafe.SliceData(c.extendedFilters)))
		c.module.XIDFC.SetFLESA(uint16(flesaPtr))
		c.module.XIDFC.SetLSE(uint8(numElements))
	}
	//***

	//*** Set up interrupts.
	// Enable interrupts.
	if c.index == 0 {
		samx51.IRQ_CAN0.EnableIRQ()
	} else {
		samx51.IRQ_CAN1.EnableIRQ()
	}

	// Enable interrupts.
	c.module.IE.SetRF0NE(true)
	c.module.IE.SetPEDE(true)
	c.module.IE.SetPEAE(true)

	// Select interrupt lines.
	c.module.ILS.SetRF0NL(true)
	c.module.ILS.SetPEDL(true)
	c.module.ILS.SetPEAL(true)

	// Enable interrupt lines.
	c.module.ILE.SetEINT0(true)
	c.module.ILE.SetEINT1(true)
	//***

	// Finally De-assert CCE bit and the CCCR.INIT bit to enable the CAN module.
	c.module.CCCR.SetCCE(false)
	c.setInit(false)

	// Copy settings from config.
	c.rx0DataLength = config.RX0DataLength
	c.rx1DataLength = config.RX1DataLength
	c.onMessageLost = config.OnMessageLost
	c.onNewMessage = config.OnNewMessage
	c.onProtocolError = config.OnProtocolError

	return nil
}

func (c *CAN) ReadFIFO(fifo FIFO, b []byte) (int, error) {
	var ptr unsafe.Pointer
	var index, fillLevel uint8
	var n int

	switch fifo {
	case 0:
		ptr = unsafe.Pointer(unsafe.SliceData(c.rx0Buffer))
		index = c.module.RXF0S.GetF0GI()
		n = FrameLengthInWords(c.rx0DataLength) * 4
		fillLevel = c.module.RXF0S.GetF0FL()
	case 1:
		ptr = unsafe.Pointer(unsafe.SliceData(c.rx1Buffer))
		index = c.module.RXF1S.GetF1GI()
		n = FrameLengthInWords(c.rx1DataLength) * 4
		fillLevel = c.module.RXF1S.GetF1FL()
	default:
		return 0, ErrFIFOOutOfRange
	}

	// Read from the FIFO if there is message available.
	if fillLevel > 0 {
		// Advance to the available message.
		ptr = unsafe.Add(ptr, n*int(index))

		// Create a slice from the resulting pointer.
		data := unsafe.Slice((*byte)(ptr), n)

		// Copy into the input buffer.
		copy(b, data)

		// Advance the index.
		c.module.RXF0A.SetF0AI(index)
	} else {
		// Nothing was read.
		n = 0
	}

	c.mutex.Unlock()
	return n, nil
}

func (c *CAN) SetFIFO(index FIFO) {
	c.mutex.Lock()
	c.fifo = index
	c.mutex.Unlock()
}

// Read will fill the input byte slice with an entire frame from the receive buffer.
func (c *CAN) Read(b []byte) (int, error) {
	return c.ReadFIFO(c.fifo, b)
}

func (c *CAN) Write(b []byte) (int, error) {
	c.mutex.Lock()
	ptr := unsafe.Pointer(unsafe.SliceData(c.txBuffer))
	n := FrameLengthInBytes(c.rx0DataLength)
	index := int(c.module.TXFQS.GetTFQPI())

	// Transmit a frame if the queue is not full.
	if !c.module.TXFQS.GetTFQF() {
		// Get the base address of the next available slot.
		ptr = unsafe.Add(ptr, n*index)
		frameBuffer := unsafe.Slice((*byte)(ptr), n)

		// Copy the data.
		copy(frameBuffer, b)

		// Advance the index.
		c.module.TXEFA.SetEFAI(uint8(index))
	} else {
		c.mutex.Unlock()
		return 0, ErrQueueFull
	}

	c.mutex.Unlock()
	return n, nil
}

func (c *CAN) SendFrame(frame Frame) error {
	c.mutex.Lock()
	ptr := unsafe.Pointer(unsafe.SliceData(c.txBuffer))
	n := FrameLengthInBytes(c.rx0DataLength)
	index := int(c.module.TXFQS.GetTFQPI())

	// Transmit a frame if the queue is not full.
	if !c.module.TXFQS.GetTFQF() {
		// Get the base address of the next available slot.
		ptr = unsafe.Add(ptr, n*index)
		frameBuffer := unsafe.Slice((*byte)(ptr), n)

		// Zero the frame buffer.
		memset(ptr, 0, uintptr(n))

		// Set the frame header.
		header := (*frameHeader)(ptr)
		header.setID(frame.ID)
		header.setDLC(uint8(dataLengthInBytes(frame.DataLengthCode)))
		header.setXTD(frame.Extended)
		header.setFDF(frame.FD)

		// Copy the data.
		copy(frameBuffer[headerLength:], frame.Data)

		// Advance the index.
		c.module.TXBAR = 1 << index
	} else {
		c.mutex.Unlock()
		return ErrQueueFull
	}

	c.mutex.Unlock()
	return nil
}

func (c *CAN) ReceiveFrame(fifo FIFO, frameBuffer []byte) (Frame, error) {
	var n int
	switch fifo {
	case 0:
		n = FrameLengthInBytes(c.rx0DataLength)
	case 1:
		n = FrameLengthInBytes(c.rx1DataLength)
	default:
		return Frame{}, ErrFIFOOutOfRange
	}

	if len(frameBuffer) < n {
		return Frame{Nil: true}, ErrFrameBufferTooSmall
	}

	// Receive the frame header data.
	if n, err := c.ReadFIFO(fifo, frameBuffer); err != nil || n <= 0 {
		return Frame{Nil: true}, err
	}

	// Decode frame header.
	ptr := unsafe.Pointer(unsafe.SliceData(frameBuffer))
	header := (*frameHeader)(ptr)

	// Return the (simple) frame.
	frame := Frame{
		ID:             header.ID(),
		Data:           frameBuffer[headerLength:],
		DataLengthCode: ByteLengthToDataLengthCode(int(header.DLC())),
		Extended:       header.XTD(),
		Nil:            false,
		FD:             header.FDF(),
		buf:            frameBuffer,
	}
	return frame, nil
}

func (c *CAN) _onNewMessage(fifo FIFO) {
	module := c.module

	if c.onNewMessage != nil {
		c.onNewMessage(fifo)
	}

	switch fifo {
	case 0:
		// Clear the interrupt.
		module.IR.SetRF0N(true)
	case 1:
		// Clear the interrupt.
		module.IR.SetRF1N(true)
	}
}

func (c *CAN) _onMessageLost(fifo FIFO) {
	module := c.module

	if c.onMessageLost != nil {
		c.onMessageLost(fifo)
	}

	switch fifo {
	case 0:
		// Clear the interrupt.
		module.IR.SetRF0L(true)
	case 1:
		// Clear the interrupt.
		module.IR.SetRF1L(true)
	}
}

func (c *CAN) _onProtocolError() {
	module := c.module

	if c.onProtocolError != nil {
		c.onProtocolError()
	}

	// Clear the interrupts.
	switch {
	case module.IR.GetPED():
		module.IR.SetPED(true)
	case module.IR.GetPEA():
		module.IR.SetPEA(true)
	}
}

func canEvent(index int8) {
	instance := can[index]
	module := instance.module

	if module.IR.GetPED() || module.IR.GetPEA() {
		instance._onProtocolError()
	}

	if module.IR.GetRF0N() {
		instance._onNewMessage(FIFO0)
	}

	if module.IR.GetRF0L() {
		instance._onMessageLost(FIFO0)
	}

	if module.IR.GetRF1N() {
		instance._onNewMessage(FIFO1)
	}

	if module.IR.GetRF1L() {
		instance._onMessageLost(FIFO1)
	}
}

func (c *CAN) setInit(on bool) {
	// Toggle the bit.
	c.module.CCCR.SetINIT(on)

	// Wait for sync...
	for c.module.CCCR.GetINIT() != on {
	}
}

//sigo:interrupt CAN0_Handler CAN0_Handler
func CAN0_Handler() {
	canEvent(0)
}

//sigo:interrupt CAN1_Handler CAN1_Handler
func CAN1_Handler() {
	canEvent(1)
}

//go:inline translateDLC0
func translateDLC0(code DataLengthCode) chip.CAN_RXESC_REG_F0DS {
	switch code {
	case DLC8:
		return chip.CAN_RXESC_REG_F0DS_DATA8
	case DLC12:
		return chip.CAN_RXESC_REG_F0DS_DATA12
	case DLC16:
		return chip.CAN_RXESC_REG_F0DS_DATA16
	case DLC20:
		return chip.CAN_RXESC_REG_F0DS_DATA20
	case DLC24:
		return chip.CAN_RXESC_REG_F0DS_DATA24
	case DLC32:
		return chip.CAN_RXESC_REG_F0DS_DATA32
	case DLC48:
		return chip.CAN_RXESC_REG_F0DS_DATA48
	case DLC64:
		return chip.CAN_RXESC_REG_F0DS_DATA64
	default:
		panic("invalid data length code")
	}
}

//go:inline translateDLC1
func translateDLC1(code DataLengthCode) chip.CAN_RXESC_REG_F1DS {
	switch code {
	case DLC8:
		return chip.CAN_RXESC_REG_F1DS_DATA8
	case DLC12:
		return chip.CAN_RXESC_REG_F1DS_DATA12
	case DLC16:
		return chip.CAN_RXESC_REG_F1DS_DATA16
	case DLC20:
		return chip.CAN_RXESC_REG_F1DS_DATA20
	case DLC24:
		return chip.CAN_RXESC_REG_F1DS_DATA24
	case DLC32:
		return chip.CAN_RXESC_REG_F1DS_DATA32
	case DLC48:
		return chip.CAN_RXESC_REG_F1DS_DATA48
	case DLC64:
		return chip.CAN_RXESC_REG_F1DS_DATA64
	default:
		panic("invalid data length code")
	}
}

//go:inline FrameLengthInWords
func FrameLengthInWords(code DataLengthCode) int {
	switch code {
	case DLC8:
		return 4
	case DLC12:
		return 5
	case DLC16:
		return 6
	case DLC20:
		return 7
	case DLC24:
		return 8
	case DLC32:
		return 10
	case DLC48:
		return 14
	case DLC64:
		return 18
	default:
		panic("invalid data length code")
	}
}

//go:inline FrameLengthInBytes
func FrameLengthInBytes(code DataLengthCode) int {
	return FrameLengthInWords(code) * 4
}
