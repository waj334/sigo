//go:build atsame5x && !generic

package can

import (
	"peripheral"
	"peripheral/pin"
	"sync"
	"unsafe"

	"runtime/arm/cortexm/sam/atsamx5x"
	"runtime/arm/cortexm/sam/atsamx5x/support/can"
)

//sigo:extern memset memset
func memset(ptr unsafe.Pointer, value int, num uintptr) unsafe.Pointer

const (
	FIFOMode  = false
	QueueMode = true

	ErrFrameBufferTooSmall Error = -1
	ErrFIFOOutOfRange      Error = -2
	ErrQueueFull           Error = -3

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
	DefaultTiming = BitTiming{
		isDefault: true,
	}
)

type Error int
type FilterMode uint8
type FIFO uint8
type FilterType uint8
type FilterConfig uint8

func (e Error) Error() string {
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

type _can struct {
	module *can.PeripheralCanType

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

	standardFilters []can.RegisterGroupCanSidfeType
	extendedFilters []can.RegisterGroupCanXidfeType
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

func (f *Filter) toStandardFilter() (result can.RegisterGroupCanSidfeType) {
	result.CanSidfe0.SetSft(can.ConstantCanSidfe0SftType(f.Type & 0x3))
	result.CanSidfe0.SetSfec(can.ConstantCanSidfe0SfecType(f.Config & 0x7))
	result.CanSidfe0.SetSfid1(uint16(f.ID1 & 0x3FF))
	result.CanSidfe0.SetSfid2(uint16(f.ID2 & 0x3FF))
	return
}

func (f *Filter) toExtendedFilter() (result can.RegisterGroupCanXidfeType) {
	result.CanXidfe0.SetEfec(can.ConstantCanXidfe0EfecType(f.Config & 0x7))
	result.CanXidfe0.SetEfid1(f.ID1 & 0xFFFFFFF)
	result.CanXidfe1.SetEft(can.ConstantCanXidfe1EftType(f.Type & 0x3))
	result.CanXidfe1.SetEfid2(f.ID2 & 0xFFFFFFF)
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
			// _can 0
			return 0, nil
		}
	} else if c.TX == pin.PB11 || c.TX == pin.PB14 {
		if c.RX == pin.PB13 || c.RX == pin.PB15 {
			// _can 1
			return 1, nil
		}
	}
	return -1, peripheral.ErrInvalidPinout
}

func (c *_can) Configure(config Config) error {
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
	c.module = can.Can[c.index]

	// Assert the Initialization (CCCR.INIT) bit so that the _can module _can be configured.
	c.setInit(true)

	// Assert the Configuration Change Enable (CCCR.CCE) bit.
	c.module.Cccr.SetCce(true)

	// Set up bit timings.
	if !config.NominalBitTiming.isDefault {
		c.module.Nbtp.SetNtseg2(config.NominalBitTiming.AfterSample)
		c.module.Nbtp.SetNtseg1(config.NominalBitTiming.BeforeSample)
		c.module.Nbtp.SetNbrp(config.NominalBitTiming.Prescaler)
		c.module.Nbtp.SetNsjw(config.NominalBitTiming.JumpWidth)
	}

	if !config.DataBitTiming.isDefault {
		c.module.Dbtp.SetDtseg2(config.DataBitTiming.AfterSample)
		c.module.Dbtp.SetDtseg1(config.DataBitTiming.BeforeSample)
		c.module.Dbtp.SetDbrp(uint8(config.DataBitTiming.Prescaler))
		c.module.Dbtp.SetDsjw(config.DataBitTiming.JumpWidth)
		c.module.Dbtp.SetTdc(config.DataBitTiming.DelayCompensation)
	}

	// Enable/disable _can FD.
	c.module.Cccr.SetFdoe(config.FD)
	c.module.Cccr.SetBrse(config.FD)

	//*** Set up transmitter
	if config.TXNumElements > 0 {
		bufLen := FrameLengthInWords(config.TXDataLength) * int(config.TXNumElements)
		c.txBuffer = make([]uint32, bufLen)
		txBufferPtr := uintptr(unsafe.Pointer(unsafe.SliceData(c.txBuffer)))

		c.module.Txesc.SetTbds(can.TxescTbdsData8)
		c.module.Txbc.SetTbsa(uint16(txBufferPtr))
		c.module.Txbc.SetNdtb(config.TXNumElements)
		c.module.Txbc.SetTfqm(config.TXQueueMode)
	}
	//***

	//*** Set up receiver.
	if config.RX0NumElements > 0 {
		c.module.Rxesc.SetF0ds(translateDLC0(config.RX0DataLength))

		// Set the receive buffer address for FIFO 0.
		bufLen := FrameLengthInWords(config.RX0DataLength) * int(config.RX0NumElements)
		c.rx0Buffer = make([]uint32, bufLen)
		rxBufferPtr := uintptr(unsafe.Pointer(unsafe.SliceData(c.rx0Buffer)))
		c.module.Rxf0c.SetF0sa(uint16(rxBufferPtr))
		c.module.Rxf0c.SetF0s(config.RX0NumElements)
	}

	if config.RX1NumElements > 0 {
		c.module.Rxesc.SetF1ds(translateDLC1(config.RX1DataLength))

		// Set the receive buffer address for FIFO 0.
		bufLen := FrameLengthInWords(config.RX1DataLength) * int(config.RX1NumElements)
		c.rx1Buffer = make([]uint32, bufLen)
		rxBufferPtr := uintptr(unsafe.Pointer(unsafe.SliceData(c.rx1Buffer)))
		c.module.Rxf1c.SetF1sa(uint16(rxBufferPtr))
		c.module.Rxf1c.SetF1s(config.RX1NumElements)
	}
	//***

	//*** Set up filters.
	c.module.Gfc.SetRrfe(false)

	var standardFilterMode can.ConstantCanGfcAnfsType
	var extendedFilterMode can.ConstantCanGfcAnfeType

	switch config.StandardFilterMode {
	case RejectAll:
		standardFilterMode = can.GfcAnfsReject
	case AcceptInFIFO0:
		standardFilterMode = can.GfcAnfsRxf0
	case AcceptInFIFO1:
		standardFilterMode = can.GfcAnfsRxf1
	default:
		return peripheral.ErrInvalidConfig
	}
	c.module.Gfc.SetAnfs(standardFilterMode)

	switch config.ExtendedFilterMode {
	case RejectAll:
		extendedFilterMode = can.GfcAnfeReject
	case AcceptInFIFO0:
		extendedFilterMode = can.GfcAnfeRxf0
	case AcceptInFIFO1:
		extendedFilterMode = can.GfcAnfeRxf1
	default:
		return peripheral.ErrInvalidConfig
	}
	c.module.Gfc.SetAnfe(extendedFilterMode)

	if numElements := len(config.StandardFilters); numElements > 0 {
		c.standardFilters = make([]can.RegisterGroupCanSidfeType, numElements)
		for i := range config.StandardFilters {
			c.standardFilters[i] = config.StandardFilters[i].toStandardFilter()
		}

		flssaPtr := uintptr(unsafe.Pointer(unsafe.SliceData(c.standardFilters)))
		c.module.Sidfc.SetFlssa(uint16(flssaPtr))
		c.module.Sidfc.SetLss(uint8(numElements))
	}

	if numElements := len(config.StandardFilters); numElements > 0 {
		c.extendedFilters = make([]can.RegisterGroupCanXidfeType, numElements)
		for i := range config.ExtendedFilters {
			c.extendedFilters[i] = config.ExtendedFilters[i].toExtendedFilter()
		}

		flesaPtr := uintptr(unsafe.Pointer(unsafe.SliceData(c.extendedFilters)))
		c.module.Xidfc.SetFlesa(uint16(flesaPtr))
		c.module.Xidfc.SetLse(uint8(numElements))
	}
	//***

	//*** Set up interrupts.
	// Enable interrupts.
	if c.index == 0 {
		atsamx5x.IRQ_CAN0.EnableIRQ()
	} else {
		atsamx5x.IRQ_CAN1.EnableIRQ()
	}

	// Enable interrupts.
	c.module.Ie.SetRf0ne(true)
	c.module.Ie.SetPede(true)
	c.module.Ie.SetPeae(true)

	// Select interrupt lines.
	c.module.Ils.SetRf0nl(true)
	c.module.Ils.SetPedl(true)
	c.module.Ils.SetPeal(true)

	// Enable interrupt lines.
	c.module.Ile.SetEint0(true)
	c.module.Ile.SetEint1(true)
	//***

	// Finally De-assert CCE bit and the CCCR.INIT bit to enable the _can module.
	c.module.Cccr.SetCce(false)
	c.setInit(false)

	// Copy settings from config.
	c.rx0DataLength = config.RX0DataLength
	c.rx1DataLength = config.RX1DataLength
	c.onMessageLost = config.OnMessageLost
	c.onNewMessage = config.OnNewMessage
	c.onProtocolError = config.OnProtocolError

	return nil
}

func (c *_can) ReadFIFO(fifo FIFO, b []byte) (int, error) {
	var ptr unsafe.Pointer
	var index, fillLevel uint8
	var n int

	switch fifo {
	case 0:
		ptr = unsafe.Pointer(unsafe.SliceData(c.rx0Buffer))
		index = c.module.Rxf0s.GetF0gi()
		n = FrameLengthInWords(c.rx0DataLength) * 4
		fillLevel = c.module.Rxf0s.GetF0fl()
	case 1:
		ptr = unsafe.Pointer(unsafe.SliceData(c.rx1Buffer))
		index = c.module.Rxf1s.GetF1gi()
		n = FrameLengthInWords(c.rx1DataLength) * 4
		fillLevel = c.module.Rxf1s.GetF1fl()
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
		c.module.Rxf0a.SetF0ai(index)
	} else {
		// Nothing was read.
		n = 0
	}

	c.mutex.Unlock()
	return n, nil
}

func (c *_can) SetFIFO(index FIFO) {
	c.mutex.Lock()
	c.fifo = index
	c.mutex.Unlock()
}

// Read will fill the input byte slice with an entire frame from the receive buffer.
func (c *_can) Read(b []byte) (int, error) {
	return c.ReadFIFO(c.fifo, b)
}

func (c *_can) Write(b []byte) (int, error) {
	c.mutex.Lock()
	ptr := unsafe.Pointer(unsafe.SliceData(c.txBuffer))
	n := FrameLengthInBytes(c.rx0DataLength)
	index := int(c.module.Txfqs.GetTfqpi())

	// Transmit a frame if the queue is not full.
	if !c.module.Txfqs.GetTfqf() {
		// Get the base address of the next available slot.
		ptr = unsafe.Add(ptr, n*index)
		frameBuffer := unsafe.Slice((*byte)(ptr), n)

		// Copy the data.
		copy(frameBuffer, b)

		// Advance the index.
		c.module.Txefa.SetEfai(uint8(index))
	} else {
		c.mutex.Unlock()
		return 0, ErrQueueFull
	}

	c.mutex.Unlock()
	return n, nil
}

func (c *_can) SendFrame(frame Frame) error {
	c.mutex.Lock()
	ptr := unsafe.Pointer(unsafe.SliceData(c.txBuffer))
	n := FrameLengthInBytes(c.rx0DataLength)
	index := int(c.module.Txfqs.GetTfqpi())

	// Transmit a frame if the queue is not full.
	if !c.module.Txfqs.GetTfqf() {
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
		c.module.Txbar = 1 << index
	} else {
		c.mutex.Unlock()
		return ErrQueueFull
	}

	c.mutex.Unlock()
	return nil
}

func (c *_can) ReceiveFrame(fifo FIFO, frameBuffer []byte) (Frame, error) {
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

func (c *_can) _onNewMessage(fifo FIFO) {
	module := c.module

	if c.onNewMessage != nil {
		c.onNewMessage(fifo)
	}

	switch fifo {
	case 0:
		// Clear the interrupt.
		module.Ir.SetRf0n(true)
	case 1:
		// Clear the interrupt.
		module.Ir.SetRf1n(true)
	}
}

func (c *_can) _onMessageLost(fifo FIFO) {
	module := c.module

	if c.onMessageLost != nil {
		c.onMessageLost(fifo)
	}

	switch fifo {
	case 0:
		// Clear the interrupt.
		module.Ir.SetRf0l(true)
	case 1:
		// Clear the interrupt.
		module.Ir.SetRf1l(true)
	}
}

func (c *_can) _onProtocolError() {
	module := c.module

	if c.onProtocolError != nil {
		c.onProtocolError()
	}

	// Clear the interrupts.
	switch {
	case module.Ir.GetPed():
		module.Ir.SetPed(true)
	case module.Ir.GetPea():
		module.Ir.SetPea(true)
	}
}

func canEvent(index int8) {
	instance := instances[index]
	module := instance.module

	if module.Ir.GetPed() || module.Ir.GetPea() {
		instance._onProtocolError()
	}

	if module.Ir.GetRf0n() {
		instance._onNewMessage(FIFO0)
	}

	if module.Ir.GetRf0l() {
		instance._onMessageLost(FIFO0)
	}

	if module.Ir.GetRf1n() {
		instance._onNewMessage(FIFO1)
	}

	if module.Ir.GetRf1l() {
		instance._onMessageLost(FIFO1)
	}
}

func (c *_can) setInit(on bool) {
	// Toggle the bit.
	c.module.Cccr.SetInit(on)

	// Wait for sync...
	for c.module.Cccr.GetInit() != on {
	}
}

//sigo:interrupt can0Handler Can0Handler
func can0Handler() {
	canEvent(0)
}

//sigo:interrupt can1Handler Can1Handler
func can1Handler() {
	canEvent(1)
}

//go:inline translateDLC
func translateDLC0(code DataLengthCode) can.ConstantCanRxescF0dsType {
	switch code {
	case DLC8:
		return can.RxescF0dsData8
	case DLC12:
		return can.RxescF0dsData12
	case DLC16:
		return can.RxescF0dsData16
	case DLC20:
		return can.RxescF0dsData20
	case DLC24:
		return can.RxescF0dsData24
	case DLC32:
		return can.RxescF0dsData32
	case DLC48:
		return can.RxescF0dsData48
	case DLC64:
		return can.RxescF0dsData64
	default:
		panic("invalid data length code")
	}
}

//go:inline translateDLC1
func translateDLC1(code DataLengthCode) can.ConstantCanRxescF1dsType {
	switch code {
	case DLC8:
		return can.RxescF1dsData8
	case DLC12:
		return can.RxescF1dsData12
	case DLC16:
		return can.RxescF1dsData16
	case DLC20:
		return can.RxescF1dsData20
	case DLC24:
		return can.RxescF1dsData24
	case DLC32:
		return can.RxescF1dsData32
	case DLC48:
		return can.RxescF1dsData48
	case DLC64:
		return can.RxescF1dsData64
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
