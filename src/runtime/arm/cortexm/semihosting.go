package cortexm

import (
	"sync"
	"unsafe"
)

var (
	Semihosting = semihosting{}
)

//sigo:extern readSemihosting runtime_arm_cortexm.readSemihosting
func readSemihosting() byte

//sigo:extern writeSemihosting runtime_arm_cortexm.writeSemihosting
func writeSemihosting(c *byte)

type semihosting struct {
	mutex sync.Mutex
}

func (s *semihosting) Read(p []byte) (n int, err error) {
	s.mutex.Lock()
	for i := range p {
		c := readSemihosting()
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
		writeSemihosting((*byte)(ptr))
	}
	s.mutex.Unlock()
	return len(p), nil
}

func (s *semihosting) WriteString(input string) (n int, err error) {
	s.mutex.Lock()
	basePtr := unsafe.StringData(input)
	for i := 0; i < len(input); i++ {
		ptr := unsafe.Add(unsafe.Pointer(basePtr), i)
		writeSemihosting((*byte)(ptr))
	}
	s.mutex.Unlock()
	return len(input), nil
}
