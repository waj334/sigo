package can

type frameHeader struct {
	word0 uint32
	word1 uint32
}

func (f *frameHeader) ID() uint32 {
	return f.word0 & 0x1FFFFFFF
}

func (f *frameHeader) setID(id uint32) {
	f.word0 = (f.word0 &^ 0x1FFFFFFF) | (id & 0x1FFFFFFF)
}

func (f *frameHeader) RTR() bool {
	return (f.word0 & (1 << 29)) != 0
}

func (f *frameHeader) setRTR(enable bool) {
	if enable {
		f.word0 = f.word0 | (1 << 29)
	} else {
		f.word0 = f.word0 &^ (1 << 29)
	}
}

func (f *frameHeader) XTD() bool {
	return (f.word0 & (1 << 30)) != 0
}

func (f *frameHeader) setXTD(enable bool) {
	if enable {
		f.word0 = f.word0 | (1 << 30)
	} else {
		f.word0 = f.word0 &^ (1 << 30)
	}
}

func (f *frameHeader) ESI() bool {
	return (f.word0 & (1 << 31)) != 0
}

func (f *frameHeader) setESI(enable bool) {
	if enable {
		f.word0 = f.word0 | (1 << 31)
	} else {
		f.word0 = f.word0 &^ (1 << 31)
	}
}

func (f *frameHeader) RXTS() uint16 {
	return uint16(f.word1 & uint32(0xFFFF))
}

func (f *frameHeader) setRXTS(ts uint16) {
	f.word1 = (f.word1 &^ uint32(0xFFFF)) | uint32(ts)
}

func (f *frameHeader) TXTS() uint16 {
	return uint16(f.word1 & uint32(0xFFFF))
}

func (f *frameHeader) setTXTS(ts uint16) {
	f.word1 = (f.word1 &^ uint32(0xFFFF)) | uint32(ts)
}

func (f *frameHeader) DLC() uint8 {
	return uint8((f.word1 & uint32(0xF<<16)) >> 16)
}

func (f *frameHeader) setDLC(length uint8) {
	f.word1 = (f.word1 &^ uint32(0xF<<16)) | (uint32(length&0xF) << 16)
}

func (f *frameHeader) BRS() bool {
	return (f.word1 & (1 << 20)) != 0
}

func (f *frameHeader) setBRS(enable bool) {
	if enable {
		f.word1 = f.word1 | (1 << 20)
	} else {
		f.word1 = f.word1 &^ (1 << 20)
	}
}

func (f *frameHeader) FDF() bool {
	return (f.word1 & (1 << 21)) != 0
}

func (f *frameHeader) setFDF(enable bool) {
	if enable {
		f.word1 = f.word1 | (1 << 21)
	} else {
		f.word1 = f.word1 &^ (1 << 21)
	}
}

func (f *frameHeader) ET() uint8 {
	return uint8((f.word1 & uint32(0x3<<22)) >> 22)
}

func (f *frameHeader) setET(event uint8) {
	f.word1 = (f.word1 &^ uint32(0x3<<22)) | (uint32(event&0x3) << 22)
}

func (f *frameHeader) FIDX() uint8 {
	return uint8((f.word1 & uint32(0x3F<<24)) >> 24)
}

func (f *frameHeader) setFDIX(marker uint8) {
	f.word1 = (f.word1 &^ uint32(0x2F<<24)) | (uint32(marker&0x2F) << 24)
}

func (f *frameHeader) ANMF() bool {
	return (f.word1 & (1 << 31)) != 0
}

func (f *frameHeader) setANMF(enable bool) {
	if enable {
		f.word1 = f.word1 | (1 << 31)
	} else {
		f.word1 = f.word1 &^ (1 << 31)
	}
}

func (f *frameHeader) MM() uint8 {
	return uint8((f.word1 & uint32(0xFF<<24)) >> 24)
}

func (f *frameHeader) setMM(marker uint8) {
	f.word1 = (f.word1 &^ uint32(0xFF<<24)) | (uint32(marker&0xFF) << 24)
}
