package abi

import "unsafe"

// Kind represents the specific kind of type that a Type represents.
// The values must match runtime.kind exactly.
type Kind uint8

const (
	Invalid Kind = iota
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
	Array
	Chan
	Func
	Interface
	Map
	Pointer
	Slice
	String
	Struct
	UnsafePointer
)

// Type is the runtime representation of a Go type.
// This struct is layout-compatible with runtime._type:
//
//	{size uintptr, data unsafe.Pointer, name string, kind uint8}
type Type struct {
	Size_ uintptr
	Data  unsafe.Pointer
	Name_ string
	Kind_ Kind
}

func (t *Type) Kind() Kind    { return t.Kind_ }
func (t *Type) Size() uintptr { return t.Size_ }

// FieldAlign returns the alignment in bytes for a value of this type
// when used as a struct field.
func (t *Type) FieldAlign() int {
	s := int(t.Size_)
	if s == 0 {
		return 1
	}
	maxAlign := int(unsafe.Sizeof(uintptr(0)))
	if s >= maxAlign {
		return maxAlign
	}
	// Largest power of 2 that fits in s.
	a := 1
	for a*2 <= s {
		a *= 2
	}
	return a
}

// iface matches the runtime interface layout: {value, valueT}.
type iface struct {
	value  unsafe.Pointer
	valueT *Type
}

// TypeOf returns the *Type for the dynamic type of the interface value v.
func TypeOf(v any) *Type {
	eface := *(*iface)(unsafe.Pointer(&v))
	return eface.valueT
}

// TypeFor returns the *Type for the type argument T.
func TypeFor[T any]() *Type {
	var zero T
	return TypeOf(zero)
}

// ---------- Array ----------

// _arrayTypeData mirrors runtime._arrayTypeData.
type _arrayTypeData struct {
	length      uint16
	elementType *Type
}

// ArrayType holds array type metadata.
type ArrayType struct {
	Elem *Type
	Len  uintptr
}

// ArrayType returns the array-specific type information.
func (t *Type) ArrayType() *ArrayType {
	d := (*_arrayTypeData)(t.Data)
	return &ArrayType{
		Elem: d.elementType,
		Len:  uintptr(d.length),
	}
}

// ---------- Struct ----------

// _structFieldData mirrors runtime._structFieldData.
type _structFieldData struct {
	dataType *Type
	tag      string
	offset   uintptr
}

// _structTypeData mirrors runtime._structTypeData.
type _structTypeData struct {
	fields []_structFieldData
}

// StructField describes a single struct field.
type StructField struct {
	Typ    *Type
	Offset uintptr
}

// StructType holds struct type metadata.
type StructType struct {
	Fields []StructField
}

// StructType returns the struct-specific type information.
func (t *Type) StructType() *StructType {
	d := (*_structTypeData)(t.Data)
	fields := make([]StructField, len(d.fields))
	for i := range d.fields {
		fields[i] = StructField{
			Typ:    d.fields[i].dataType,
			Offset: d.fields[i].offset,
		}
	}
	return &StructType{Fields: fields}
}

// ---------- Map ----------

// _mapTypeData mirrors runtime._mapTypeData.
type _mapTypeData struct {
	keyType     *Type
	elementType *Type
}

// MapType holds map type metadata.
type MapType struct {
	Key    *Type
	Elem   *Type
	Hasher func(unsafe.Pointer, uintptr) uintptr
}

// MapType returns the map-specific type information.
func (t *Type) MapType() *MapType {
	d := (*_mapTypeData)(t.Data)
	return &MapType{
		Key:    d.keyType,
		Elem:   d.elementType,
		Hasher: makeHasher(d.keyType),
	}
}

// ---------- Hasher ----------

const (
	fnvBasis uint64 = 0xcbf29ce484222325
	fnvPrime uint64 = 0x100000001b3
)

// _string mirrors the runtime string layout.
type _string struct {
	data unsafe.Pointer
	len  int
}

func computeFnvSeeded(ptr unsafe.Pointer, size uintptr, seed uintptr) uintptr {
	hash := fnvBasis ^ uint64(seed)
	for i := uintptr(0); i < size; i++ {
		b := *(*byte)(unsafe.Add(ptr, i))
		hash = hash * fnvPrime
		hash = hash ^ uint64(b)
	}
	return uintptr(hash)
}

func makeHasher(keyType *Type) func(unsafe.Pointer, uintptr) uintptr {
	switch keyType.Kind() {
	case String:
		return func(p unsafe.Pointer, seed uintptr) uintptr {
			s := (*_string)(p)
			return computeFnvSeeded(s.data, uintptr(s.len), seed)
		}
	case Array:
		d := (*_arrayTypeData)(keyType.Data)
		size := uintptr(d.elementType.Size_) * uintptr(d.length)
		return func(p unsafe.Pointer, seed uintptr) uintptr {
			return computeFnvSeeded(p, size, seed)
		}
	default:
		size := keyType.Size_
		return func(p unsafe.Pointer, seed uintptr) uintptr {
			return computeFnvSeeded(p, size, seed)
		}
	}
}
