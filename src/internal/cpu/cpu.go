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

// Included for compatibility ---

var X86 struct {
	_                   CacheLinePad
	HasAES              bool
	HasADX              bool
	HasAVX              bool
	HasAVXVNNI          bool
	HasAVX2             bool
	HasAVX512           bool // Virtual feature: F+CD+BW+DQ+VL
	HasAVX512F          bool
	HasAVX512CD         bool
	HasAVX512BW         bool
	HasAVX512DQ         bool
	HasAVX512VL         bool
	HasAVX512GFNI       bool
	HasAVX512VAES       bool
	HasAVX512VNNI       bool
	HasAVX512VBMI       bool
	HasAVX512VBMI2      bool
	HasAVX512BITALG     bool
	HasAVX512VPOPCNTDQ  bool
	HasAVX512VPCLMULQDQ bool
	HasBMI1             bool
	HasBMI2             bool
	HasERMS             bool
	HasFSRM             bool
	HasFMA              bool
	HasGFNI             bool
	HasOSXSAVE          bool
	HasPCLMULQDQ        bool
	HasPOPCNT           bool
	HasRDTSCP           bool
	HasSHA              bool
	HasSSE3             bool
	HasSSSE3            bool
	HasSSE41            bool
	HasSSE42            bool
	HasVAES             bool
	_                   CacheLinePad
}

var ARM64 struct {
	_          CacheLinePad
	HasAES     bool
	HasPMULL   bool
	HasSHA1    bool
	HasSHA2    bool
	HasSHA512  bool
	HasSHA3    bool
	HasCRC32   bool
	HasATOMICS bool
	HasCPUID   bool
	HasDIT     bool
	IsNeoverse bool
	_          CacheLinePad
}

var Loong64 struct {
	_         CacheLinePad
	HasLSX    bool // support 128-bit vector extension
	HasLASX   bool // support 256-bit vector extension
	HasCRC32  bool // support CRC instruction
	HasLAMCAS bool // support AMCAS[_DB].{B/H/W/D}
	HasLAM_BH bool // support AM{SWAP/ADD}[_DB].{B/H} instruction
	_         CacheLinePad
}

var MIPS64X struct {
	_      CacheLinePad
	HasMSA bool // MIPS SIMD architecture
	_      CacheLinePad
}

var PPC64 struct {
	_         CacheLinePad
	HasDARN   bool // Hardware random number generator (requires kernel enablement)
	HasSCV    bool // Syscall vectored (requires kernel enablement)
	IsPOWER8  bool // ISA v2.07 (POWER8)
	IsPOWER9  bool // ISA v3.00 (POWER9)
	IsPOWER10 bool // ISA v3.1  (POWER10)
	_         CacheLinePad
}

var S390X struct {
	_         CacheLinePad
	HasZARCH  bool // z architecture mode is active [mandatory]
	HasSTFLE  bool // store facility list extended [mandatory]
	HasLDISP  bool // long (20-bit) displacements [mandatory]
	HasEIMM   bool // 32-bit immediates [mandatory]
	HasDFP    bool // decimal floating point
	HasETF3EH bool // ETF-3 enhanced
	HasMSA    bool // message security assist (CPACF)
	HasAES    bool // KM-AES{128,192,256} functions
	HasAESCBC bool // KMC-AES{128,192,256} functions
	HasAESCTR bool // KMCTR-AES{128,192,256} functions
	HasAESGCM bool // KMA-GCM-AES{128,192,256} functions
	HasGHASH  bool // KIMD-GHASH function
	HasSHA1   bool // K{I,L}MD-SHA-1 functions
	HasSHA256 bool // K{I,L}MD-SHA-256 functions
	HasSHA512 bool // K{I,L}MD-SHA-512 functions
	HasSHA3   bool // K{I,L}MD-SHA3-{224,256,384,512} and K{I,L}MD-SHAKE-{128,256} functions
	HasVX     bool // vector facility. Note: the runtime sets this when it processes auxv records.
	HasVXE    bool // vector-enhancements facility 1
	HasKDSA   bool // elliptic curve functions
	HasECDSA  bool // NIST curves
	HasEDDSA  bool // Edwards curves
	_         CacheLinePad
}

var RISCV64 struct {
	_                 CacheLinePad
	HasFastMisaligned bool // Fast misaligned accesses
	HasV              bool // Vector extension compatible with RVV 1.0
	HasZbb            bool // Basic bit-manipulation extension
	_                 CacheLinePad
}
