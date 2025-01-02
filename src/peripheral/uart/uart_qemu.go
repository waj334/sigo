//go:build qemu_cortexm4

package uart

import (
	"peripheral"
	"runtime/arm/cortexm/qemu/chip"
	"runtime/ringbuffer"
	"sync"
)

type UART struct {
	i        uint8
	txBuffer ringbuffer.RingBuffer
	rxBuffer ringbuffer.RingBuffer
	mutex    sync.Mutex
}

type Config struct {
	BaudHz          uint16
	CharacterSize   uint8
	NumStopBits     uint8
	TXBufferSize    uintptr
	RXBufferSize    uintptr
	ReceiveEnabled  bool
	TransmitEnabled bool
}

func (u *UART) Configure(config Config) error {
	var wordLength bool
	if config.CharacterSize == 8 {
		wordLength = false
	} else if config.CharacterSize == 9 {
		wordLength = true
	} else {
		return peripheral.ErrInvalidConfig
	}

	if config.NumStopBits > 3 {
		return peripheral.ErrInvalidConfig
	}

	usart := chip.USART[u.i]

	// Enable the USART.
	usart.CR1.SetUE(true)

	// Define the word length.
	usart.CR1.SetM(wordLength)

	// Set the number of stop bits.
	usart.CR2.SetSTOP(config.NumStopBits)

	// Calculate the baud rate.

	return nil
}

func (u *UART) WriteString(s string) (n int, err error) {
	//TODO implement me
	panic("implement me")
}

func (u *UART) Read(p []byte) (n int, err error) {
	//TODO implement me
	panic("implement me")
}

func (u *UART) Write(p []byte) (n int, err error) {
	//TODO implement me
	panic("implement me")
}
