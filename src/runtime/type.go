package runtime

import "unsafe"

// BasicKind describes the kind of basic type.
type BasicKind int

const (
	InvalidBasicKind BasicKind = iota // type is invalid

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

type ConstructType int

const (
	InvalidConstructType ConstructType = iota
	Primitive
	Pointer
	Interface
	Struct
	Array
	Slice
	Map
	Channel
)

type _type struct {
	name      *string
	size      uintptr
	construct ConstructType
	kind      BasicKind
	methods   *_methodTable
	fields    *_fieldTable
	array     *_arrayType
	mapp      *_mapType
	ptr       *_pointerType
	channel   *_channelType
	function  *_funcType
}

type _methodTable struct {
	count   int
	methods unsafe.Pointer
}

func (m _methodTable) index(i int) *_funcType {
	if i < m.count {
		ptr := unsafe.Add(m.methods, unsafe.Sizeof(uintptr(0))*uintptr(i))
		return (*_funcType)(unsafe.Pointer(*(*uintptr)(ptr)))
	}
	return nil
}

type _fieldTable struct {
	count  int
	fields unsafe.Pointer
}

func (f _fieldTable) index(i int) *_field {
	if i < f.count {
		ptr := unsafe.Add(f.fields, unsafe.Sizeof(uintptr(0))*uintptr(i))
		return (*_field)(unsafe.Pointer(*(*uintptr)(ptr)))
	}
	return nil
}

type _typeTable struct {
	count int
	types unsafe.Pointer
}

func (t _typeTable) index(i int) *_type {
	if i < t.count {
		ptr := unsafe.Add(t.types, unsafe.Sizeof(uintptr(0))*uintptr(i))
		return (*_type)(unsafe.Pointer(*(*uintptr)(ptr)))
	}
	return nil
}

type _funcType struct {
	ptr     unsafe.Pointer
	id      uint32
	name    *string
	args    *_typeTable
	returns *_typeTable
}

type _field struct {
	name     *string
	typeInfo *_type
	tag      *string
}

type _arrayType struct {
	elementType *_type
	length      int
}

type _mapType struct {
	keyType   *_type
	valueType *_type
}

type _pointerType struct {
	elementType *_type
}

type _channelType struct {
	elementType *_type
	direction   int
}
