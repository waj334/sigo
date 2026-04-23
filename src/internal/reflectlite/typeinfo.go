package reflectlite

import "unsafe"

// _type matches the runtime's type descriptor layout exactly.
// Field order must match runtime._type: {size, data, name, kind}.
type _type struct {
	size uintptr
	data unsafe.Pointer
	name string
	kind uint8
}

// toType extracts the underlying *_type from a Type interface value.
func toType(t Type) *_type {
	eface := (*_interface)(unsafe.Pointer(&t))
	return (*_type)(eface.value)
}

func (t *_type) Name() string {
	if len(t.name) > 0 {
		return t.name
	}
	return ""
}

func (t *_type) PkgPath() string {
	return ""
}

func (t *_type) Size() uintptr {
	return t.size
}

func (t *_type) Kind() Kind {
	return Kind(t.kind)
}

func (t *_type) Implements(u Type) bool {
	if u == nil {
		panic("reflect: nil type passed to Type.Implements")
	}
	if u.Kind() != Interface {
		panic("reflect: non-interface type passed to Type.Implements")
	}
	return typeImplements(t, toType(u))
}

// typeImplements reports whether type t implements interface type u.
// This mirrors the logic in runtime.interfaceIsAssignable.
func typeImplements(t *_type, u *_type) bool {
	interfaceInfo := (*_interfaceData)(u.data)

	// Empty interface -- everything implements it.
	if len(interfaceInfo.methods) == 0 {
		return true
	}

	// Collect the methods of t.
	var tmethods []*_funcData

	// If t is a pointer, follow through to the named type.
	src := t
	if Kind(src.kind) == Pointer {
		src = (*_type)(src.data)
	}

	if len(src.name) > 0 {
		namedType := (*_namedTypeData)(src.data)
		tmethods = namedType.methods
	} else {
		// Unnamed non-interface types have no methods.
		return false
	}

	// Every method in the interface must exist in t's methods (by id).
	for _, um := range interfaceInfo.methods {
		found := false
		for _, tm := range tmethods {
			if um.id == tm.id {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

func (t *_type) AssignableTo(u Type) bool {
	if u == nil {
		panic("reflect: nil type passed to Type.AssignableTo")
	}
	utype := toType(u)
	if t == utype {
		return true
	}
	if u.Kind() == Interface {
		return typeImplements(t, utype)
	}
	return false
}

func (t *_type) Comparable() bool {
	switch Kind(t.kind) {
	case Slice, Map, Func:
		return false
	case Array:
		tt := (*_arrayTypeData)(t.data)
		return tt.elementType.Comparable()
	case Struct:
		tt := (*_structTypeData)(t.data)
		for _, f := range tt.fields {
			if !f.dataType.Comparable() {
				return false
			}
		}
		return true
	default:
		return true
	}
}

func (t *_type) String() string {
	if len(t.name) > 0 {
		return t.name
	}
	switch Kind(t.kind) {
	case Invalid:
		return "invalid"
	case Bool:
		return "bool"
	case Int:
		return "int"
	case Int8:
		return "int8"
	case Int16:
		return "int16"
	case Int32:
		return "int32"
	case Int64:
		return "int64"
	case Uint:
		return "uint"
	case Uint8:
		return "uint8"
	case Uint16:
		return "uint16"
	case Uint32:
		return "uint32"
	case Uint64:
		return "uint64"
	case Uintptr:
		return "uintptr"
	case Float32:
		return "float32"
	case Float64:
		return "float64"
	case Complex64:
		return "complex64"
	case Complex128:
		return "complex128"
	case Array:
		tt := (*_arrayTypeData)(t.data)
		return "[" + itoa(int(tt.length)) + "]" + tt.elementType.String()
	case Chan:
		tt := (*_channelTypeData)(t.data)
		switch tt.direction {
		case 1: // RecvDir
			return "<-chan " + tt.elementType.String()
		case 2: // SendDir
			return "chan<- " + tt.elementType.String()
		default:
			return "chan " + tt.elementType.String()
		}
	case Func:
		return "func(...)"
	case Interface:
		return "interface {}"
	case Map:
		tt := (*_mapTypeData)(t.data)
		return "map[" + tt.keyType.String() + "]" + tt.elementType.String()
	case Pointer:
		elem := (*_type)(t.data)
		return "*" + elem.String()
	case Slice:
		elem := (*_type)(t.data)
		return "[]" + elem.String()
	case String:
		return "string"
	case Struct:
		return "struct { ... }"
	case UnsafePointer:
		return "unsafe.Pointer"
	}
	return "unknown"
}

func (t *_type) Elem() Type {
	switch Kind(t.kind) {
	case Array:
		tt := (*_arrayTypeData)(t.data)
		return tt.elementType
	case Chan:
		tt := (*_channelTypeData)(t.data)
		return tt.elementType
	case Map:
		tt := (*_mapTypeData)(t.data)
		return tt.elementType
	case Pointer:
		return (*_type)(t.data)
	case Slice:
		return (*_type)(t.data)
	}
	panic("reflect: Elem of invalid type " + t.String())
}

// itoa converts a non-negative int to its decimal string representation.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
