package cortexm

import (
	"asm"
	"asm/register"
	"sync"
	"unsafe"
)

var (
	Semihosting = semihosting{}
)

type semihosting struct {
	mutex sync.Mutex
}

func (s *semihosting) Read(p []byte) (n int, err error) {
	s.mutex.Lock()
	for i := range p {
		var c byte
		asm.Inline(`
			mov r0, #0x07
			bkpt 0xAB
		`, asm.Out(&c))
		if c == '\000' {
			s.mutex.Unlock()
			return i, nil
		}
		p[i] = c
	}
	s.mutex.Unlock()
	return len(p), nil
}

func (s *semihosting) Write(p []byte) (n int, err error) {
	s.mutex.Lock()
	basePtr := unsafe.SliceData(p)
	for i := 0; i < len(p); i++ {
		ptr := unsafe.Add(unsafe.Pointer(basePtr), i)
		asm.Inline(`
			mov r0, #0x03
			mov r1, {{ptr}}
			bkpt 0xAB
		`, asm.In(ptr), asm.Clobber(register.R0), asm.Clobber(register.R1))
	}
	s.mutex.Unlock()
	return len(p), nil
}

func (s *semihosting) WriteString(input string) (n int, err error) {
	s.mutex.Lock()
	basePtr := unsafe.StringData(input)
	for i := 0; i < len(input); i++ {
		ptr := unsafe.Add(unsafe.Pointer(basePtr), i)
		asm.Inline(`
			mov r0, #0x03
			mov r1, {{ptr}}
			bkpt 0xAB
		`, asm.In(ptr), asm.Clobber(register.R0), asm.Clobber(register.R1))
	}
	s.mutex.Unlock()
	return len(input), nil
}
