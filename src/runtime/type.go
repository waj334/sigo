package runtime

import "unsafe"

// BasicKind describes the kind of basic type.
type BasicKind int

const (
	Invalid BasicKind = iota // type is invalid

	// predeclared types
	Bool
	Int
	Int8
	Int16
	Int32
	Int64
	Uint
	Uint8
	Uint16
	Uint32
	Uint64
	Uintptr
	Float32
	Float64
	Complex64
	Complex128
	String
	UnsafePointer

	// types for untyped values
	UntypedBool
	UntypedInt
	UntypedRune
	UntypedFloat
	UntypedComplex
	UntypedString
	UntypedNil

	// aliases
	Byte = Uint8
	Rune = Int32
)

type typeDescriptor struct {
	name     *string
	size     int64
	kind     BasicKind
	methods  *methodTable
	fields   *fieldTable
	array    *arrayDescriptor
	mapp     *mapTypeDescriptor
	ptr      *pointerDescriptor
	channel  *channelTypeDescriptor
	function *functionDescriptor
}

type methodTable struct {
	count   int
	methods unsafe.Pointer
}

func (m methodTable) index(i int) *functionDescriptor {
	if i < m.count {
		return (*functionDescriptor)(unsafe.Pointer(uintptr(m.methods) + (unsafe.Sizeof(unsafe.Pointer(nil)))*uintptr(i)))
	}
	return nil
}

type fieldTable struct {
	count  int
	fields unsafe.Pointer
}

func (f fieldTable) index(i int) *fieldDescriptor {
	if i < f.count {
		return (*fieldDescriptor)(unsafe.Pointer(uintptr(f.fields) + (unsafe.Sizeof(unsafe.Pointer(nil)))*uintptr(i)))
	}
	return nil
}

type typeTable struct {
	count int
	types unsafe.Pointer
}

func (t typeTable) index(i int) *typeDescriptor {
	if i < t.count {
		return (*typeDescriptor)(unsafe.Pointer(uintptr(t.types) + (unsafe.Sizeof(unsafe.Pointer(nil)))*uintptr(i)))
	}
	return nil
}

type functionDescriptor struct {
	name    *string
	args    *typeTable
	returns *typeTable
}

type fieldDescriptor struct {
	name     *string
	typeInfo *typeDescriptor
	tag      *string
}

type arrayDescriptor struct {
	elementType *typeDescriptor
	length      int64
	capacity    int64
}

type mapTypeDescriptor struct {
	keyType   *typeDescriptor
	valueType *typeDescriptor
}

type pointerDescriptor struct {
	elementType *typeDescriptor
}

type channelTypeDescriptor struct {
	elementType *typeDescriptor
	direction   int
	capacity    int
}
