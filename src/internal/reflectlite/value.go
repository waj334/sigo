package reflectlite

import "unsafe"

const (
	flagKindMask flag = 1<<5 - 1
	flagAddr     flag = 1 << 5
	flagIndir    flag = 1 << 6
)

type flag uintptr

type Value struct {
	typ  *_type
	ptr  unsafe.Pointer
	flag flag
}

// ValueOf returns a new Value initialized to the concrete value stored in i.
// ValueOf(nil) returns the zero Value.
func ValueOf(i any) Value {
	if i == nil {
		return Value{}
	}
	eface := *(*_interface)(unsafe.Pointer(&i))
	if eface.valueT == nil {
		return Value{}
	}
	k := flag(eface.valueT.kind) & flagKindMask
	return Value{
		typ:  eface.valueT,
		ptr:  eface.value,
		flag: flagIndir | k,
	}
}

// IsValid reports whether v represents a value. It returns false if v is the zero Value.
func (v Value) IsValid() bool {
	return v.typ != nil
}

// Kind returns v's Kind.
// If v is the zero Value (IsValid returns false), Kind returns Invalid.
func (v Value) Kind() Kind {
	return Kind(v.flag & flagKindMask)
}

// Type returns v's type.
func (v Value) Type() Type {
	if v.typ == nil {
		panic("reflect: call of Value.Type on zero Value")
	}
	return v.typ
}

// Elem returns the value that the interface v contains or that the pointer v points to.
// It panics if v's Kind is not Interface or Pointer.
func (v Value) Elem() Value {
	k := v.Kind()
	switch k {
	case Interface:
		var eface _interface
		if v.flag&flagIndir != 0 {
			eface = *(*_interface)(v.ptr)
		} else {
			eface = *(*_interface)(unsafe.Pointer(&v.ptr))
		}
		if eface.valueT == nil {
			return Value{}
		}
		ek := flag(eface.valueT.kind) & flagKindMask
		return Value{
			typ:  eface.valueT,
			ptr:  eface.value,
			flag: flagIndir | ek,
		}
	case Pointer:
		var ptr unsafe.Pointer
		if v.flag&flagIndir != 0 {
			ptr = *(*unsafe.Pointer)(v.ptr)
		} else {
			ptr = v.ptr
		}
		if ptr == nil {
			return Value{}
		}
		elem := (*_type)(v.typ.data)
		ek := flag(elem.kind) & flagKindMask
		return Value{
			typ:  elem,
			ptr:  ptr,
			flag: flagIndir | flagAddr | ek,
		}
	}
	panic(&ValueError{"Value.Elem", k})
}

// IsNil reports whether its argument v is nil.
func (v Value) IsNil() bool {
	switch v.Kind() {
	case Chan, Func, Map, Pointer, UnsafePointer:
		if v.flag&flagIndir != 0 {
			return *(*unsafe.Pointer)(v.ptr) == nil
		}
		return v.ptr == nil
	case Interface:
		var eface _interface
		if v.flag&flagIndir != 0 {
			eface = *(*_interface)(v.ptr)
		} else {
			eface = *(*_interface)(unsafe.Pointer(&v.ptr))
		}
		return eface.valueT == nil
	case Slice:
		sl := (*_slice)(v.ptr)
		return sl.array == nil
	}
	panic(&ValueError{"Value.IsNil", v.Kind()})
}

// Len returns v's length.
func (v Value) Len() int {
	switch v.Kind() {
	case Array:
		tt := (*_arrayTypeData)(v.typ.data)
		return int(tt.length)
	case Slice:
		sl := (*_slice)(v.ptr)
		return sl.len
	case String:
		s := (*_string)(v.ptr)
		return s.len
	}
	panic(&ValueError{"Value.Len", v.Kind()})
}

// CanSet reports whether the value of v can be changed.
func (v Value) CanSet() bool {
	return v.flag&flagAddr != 0
}

// Set assigns x to the value v.
// It panics if CanSet returns false.
func (v Value) Set(x Value) {
	if !v.CanSet() {
		panic("reflect: Value.Set using value obtained using unexported field")
	}
	if v.typ != x.typ {
		panic("reflect: Value.Set using value of wrong type")
	}
	// Copy x.size bytes from x.ptr to v.ptr.
	size := v.typ.size
	dst := (*[1 << 20]byte)(v.ptr)
	var src *[1 << 20]byte
	if x.flag&flagIndir != 0 {
		src = (*[1 << 20]byte)(x.ptr)
	} else {
		src = (*[1 << 20]byte)(unsafe.Pointer(&x.ptr))
	}
	for i := uintptr(0); i < size; i++ {
		dst[i] = src[i]
	}
}
