package goarch

// ArchFamilyType represents a family of one or more related architectures.
type ArchFamilyType int

const (
	ARM ArchFamilyType = iota
	RISCV
	RISCV64
)

// PtrSize is the size of a pointer in bytes - unsafe.Sizeof(uintptr(0)) but as an ideal constant.
// It is also the size of the machine's native word size (that is, 4 on 32-bit systems, 8 on 64-bit).
const PtrSize = 4 << (^uintptr(0) >> 63)

// PtrBits is bit width of a pointer.
const PtrBits = PtrSize * 8

// ArchFamily is the architecture family (ARM, RISCV ...)
const ArchFamily ArchFamilyType = _ArchFamily

// BigEndian reports whether the architecture is big-endian.
const BigEndian = false

// DefaultPhysPageSize is the default physical page size.
const DefaultPhysPageSize = _DefaultPhysPageSize

// PCQuantum is the minimal unit for a program counter (1 on x86, 4 on most other systems).
// The various PC tables record PC deltas pre-divided by PCQuantum.
const PCQuantum = _PCQuantum

// Int64Align is the required alignment for a 64-bit integer (4 on 32-bit systems, 8 on 64-bit).
const Int64Align = PtrSize

// MinFrameSize is the size of the system-reserved words at the bottom
// of a frame (just above the architectural stack pointer).
const MinFrameSize = _MinFrameSize

// StackAlign is the required alignment of the SP register.
// The stack must be at least word aligned, but some architectures require more.
const StackAlign = _StackAlign
