package cpu

const CacheLinePadSize = 32

type CacheLinePad struct{ _ [CacheLinePadSize]byte }

var CacheLineSize uintptr = CacheLinePadSize

var ARM = struct {
	_            CacheLinePad
	HasVFPv4     bool
	HasIDIVA     bool
	HasV7Atomics bool
	_            CacheLinePad
}{
	HasVFPv4:     hasVFPv4,
	HasIDIVA:     hasIDIVA,
	HasV7Atomics: hasV7Atomics,
}

func Initialize(env string) {}
func Name() string          { return "" }
