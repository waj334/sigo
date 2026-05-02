package runtime

import (
	"unsafe"
)

type _interface struct {
	value  unsafe.Pointer
	valueT *_type
}

func interfaceMake(value unsafe.Pointer, valueType *_type) _interface {
	return _interface{
		value:  value,
		valueT: valueType,
	}
}

func interfaceAssert(X _interface, T *_type, hasOk bool) (result unsafe.Pointer, load bool, ok bool) {
	err := interfaceIsAssignable(X.valueT, T)
	if err != nil {
		if hasOk {
			return nil, false, false
		}
		panic(err)
	}

	// No load is necessary if the interface is already of the correct type.
	if X.valueT == T {
		return X.value, false, true
	}

	// No load is necessary if the value is a pointer type — the interface
	// value slot holds the pointer directly rather than a pointer-to-value.
	// This covers *T, unsafe.Pointer, and any type stored by reference
	// (i.e. types whose size exceeds pointer size and were heap-allocated
	// when the interface was constructed).
	if T.kind == Pointer || T.kind == UnsafePointer {
		return X.value, false, true
	}

	return X.value, true, true
}

func interfaceValue(X _interface) unsafe.Pointer {
	return X.value
}

func interfaceCompare(X _interface, Y _interface) bool {
	return interfaceCompareTo(X, Y.valueT, Y.value)
}

func interfaceCompareTo(X _interface, otherType *_type, otherValue unsafe.Pointer) bool {
	// Nil comparison
	if X.valueT == otherType && X.value == nil && otherValue == nil {
		return true
	}

	// Interfaces are equal if their types are the same and their values are the same
	if X.valueT == otherType {
		switch X.valueT.kind {
		case Bool:
			return *(*bool)(X.value) == *(*bool)(otherValue)
		case Int:
			return *(*int)(X.value) == *(*int)(otherValue)
		case Int8:
			return *(*int8)(X.value) == *(*int8)(otherValue)
		case Int16:
			return *(*int16)(X.value) == *(*int16)(otherValue)
		case Int32:
			return *(*int32)(X.value) == *(*int32)(otherValue)
		case Int64:
			return *(*int64)(X.value) == *(*int64)(otherValue)
		case Uint:
			return *(*uint)(X.value) == *(*uint)(otherValue)
		case Uint8:
			return *(*uint8)(X.value) == *(*uint8)(otherValue)
		case Uint16:
			return *(*uint16)(X.value) == *(*uint16)(otherValue)
		case Uint32:
			return *(*uint32)(X.value) == *(*uint32)(otherValue)
		case Uint64:
			return *(*uint64)(X.value) == *(*uint64)(otherValue)
		case Uintptr:
			return *(*uintptr)(X.value) == *(*uintptr)(otherValue)
		case Float32:
			return *(*float32)(X.value) == *(*float32)(otherValue)
		case Float64:
			return *(*float64)(X.value) == *(*float64)(otherValue)
		case Complex64:
			return *(*complex64)(X.value) == *(*complex64)(otherValue)
		case Complex128:
			return *(*complex128)(X.value) == *(*complex128)(otherValue)
		case String:
			return *(*string)(X.value) == *(*string)(otherValue)
		case Pointer, UnsafePointer:
			// The interface value slot holds the pointer directly.
			return X.value == otherValue
		case Chan:
			// _channel is boxed; two channels are equal iff they share state.
			return (*_channel)(X.value).state == (*_channel)(otherValue).state
		case Interface:
			// Boxed _interface; recurse.
			x := *(*_interface)(X.value)
			y := *(*_interface)(otherValue)
			return interfaceCompareTo(x, y.valueT, y.value)
		case Array:
			ad := arrayTypeData(X.valueT)
			elemSize := ad.elementType.size
			for i := uintptr(0); i < uintptr(ad.length); i++ {
				off := i * elemSize
				if !fieldEqual(ad.elementType,
					unsafe.Add(X.value, off), unsafe.Add(otherValue, off)) {
					return false
				}
			}
			return true
		case Struct:
			sd := structTypeData(X.valueT)
			for i := range sd.fields {
				f := &sd.fields[i]
				if !fieldEqual(f.dataType,
					unsafe.Add(X.value, f.offset), unsafe.Add(otherValue, f.offset)) {
					return false
				}
			}
			return true
		case Map, Slice, Func:
			panic("runtime error: comparing uncomparable type")
		}
	}
	return false
}

// fieldEqual compares two typed values whose addresses point at storage of T,
// regardless of T's kind. Unlike interfaceCompareTo, pointer-kind values here
// are stored at *(a) / *(b) rather than being a / b themselves — because they
// sit inside an enclosing array or struct rather than the interface's value
// slot.
func fieldEqual(T *_type, a, b unsafe.Pointer) bool {
	switch T.kind {
	case Bool:
		return *(*bool)(a) == *(*bool)(b)
	case Int:
		return *(*int)(a) == *(*int)(b)
	case Int8:
		return *(*int8)(a) == *(*int8)(b)
	case Int16:
		return *(*int16)(a) == *(*int16)(b)
	case Int32:
		return *(*int32)(a) == *(*int32)(b)
	case Int64:
		return *(*int64)(a) == *(*int64)(b)
	case Uint:
		return *(*uint)(a) == *(*uint)(b)
	case Uint8:
		return *(*uint8)(a) == *(*uint8)(b)
	case Uint16:
		return *(*uint16)(a) == *(*uint16)(b)
	case Uint32:
		return *(*uint32)(a) == *(*uint32)(b)
	case Uint64:
		return *(*uint64)(a) == *(*uint64)(b)
	case Uintptr:
		return *(*uintptr)(a) == *(*uintptr)(b)
	case Float32:
		return *(*float32)(a) == *(*float32)(b)
	case Float64:
		return *(*float64)(a) == *(*float64)(b)
	case Complex64:
		return *(*complex64)(a) == *(*complex64)(b)
	case Complex128:
		return *(*complex128)(a) == *(*complex128)(b)
	case String:
		return *(*string)(a) == *(*string)(b)
	case Pointer, UnsafePointer:
		return *(*unsafe.Pointer)(a) == *(*unsafe.Pointer)(b)
	case Chan:
		return (*(*_channel)(a)).state == (*(*_channel)(b)).state
	case Interface:
		x := *(*_interface)(a)
		y := *(*_interface)(b)
		return interfaceCompareTo(x, y.valueT, y.value)
	case Array:
		ad := arrayTypeData(T)
		elemSize := ad.elementType.size
		for i := uintptr(0); i < uintptr(ad.length); i++ {
			off := i * elemSize
			if !fieldEqual(ad.elementType,
				unsafe.Add(a, off), unsafe.Add(b, off)) {
				return false
			}
		}
		return true
	case Struct:
		sd := structTypeData(T)
		for i := range sd.fields {
			f := &sd.fields[i]
			if !fieldEqual(f.dataType,
				unsafe.Add(a, f.offset), unsafe.Add(b, f.offset)) {
				return false
			}
		}
		return true
	case Map, Slice, Func:
		panic("runtime error: comparing uncomparable type")
	}
	return false
}

func arrayTypeData(T *_type) *_arrayTypeData {
	if len(T.name) > 0 {
		return (*_arrayTypeData)(((*_namedTypeData)(T.data)).underlyingType.data)
	}
	return (*_arrayTypeData)(T.data)
}

func structTypeData(T *_type) *_structTypeData {
	if len(T.name) > 0 {
		return (*_structTypeData)(((*_namedTypeData)(T.data)).underlyingType.data)
	}
	return (*_structTypeData)(T.data)
}

func interfaceLookUp(i _interface, id uint32) (receiver, result unsafe.Pointer) {
	T := i.valueT

	if T.kind == Pointer {
		T = (*_type)(T.data)
	}

	elementType := (*_namedTypeData)(T.data)

	for _, method := range elementType.methods {
		if method.id == id {
			sig := method.signature

			switch {
			case i.valueT.kind == Pointer && sig.receiverType.kind != Pointer:
				// Interface holds *T, method wants T.
				// Dereference to get pointer to the actual value.
				receiver = *(*unsafe.Pointer)(i.value)

			case i.valueT.kind != Pointer && sig.receiverType.kind != Pointer:
				// Interface holds T, method wants T.
				// i.value already points to the boxed value.
				receiver = i.value

			case i.valueT.kind != Pointer && sig.receiverType.kind == Pointer:
				// Interface holds T, method wants *T.
				// -> Pass pointer to boxed value.
				receiver = i.value

			case i.valueT.kind == Pointer && sig.receiverType.kind == Pointer:
				// Interface holds *T, method wants *T.
				// -> Pass as-is.
				receiver = i.value
			}

			return receiver, method.funcPtr
		}
	}

	panic("no concrete implementation found")
}

// interfaceMethods returns the method list of an Interface-kind _type. It
// handles two layouts emitted by sigoc:
//
//   - Literal interfaces: T.data points at _interfaceData directly.
//   - Named interfaces (e.g. io.WriterTo): T.data points at _namedTypeData,
//     whose underlyingType is the *_type for the unnamed interface — and
//     that _type's data is the actual _interfaceData.
//
// Without this, the named-interface case lets the runtime read the
// _namedTypeData's underlyingType pointer as if it were the methods slice
// header, which makes every named interface look empty (matching the `any`
// fast path) and causes every interface assertion to spuriously succeed.
//
// TODO(compiler): TypeInfo.cxx's NamedType case wraps interfaces in a
// _namedTypeData when the underlying type is an InterfaceType. It would be
// cleaner to set _type.data to the underlying interface's _interfaceData
// directly so this dereference isn't needed.
func interfaceMethods(T *_type) []*_interfaceMethodData {
	if T.kind != Interface {
		return nil
	}
	if len(T.name) > 0 {
		// Named interface: data is _namedTypeData; follow underlyingType
		// to its _interfaceData.
		named := (*_namedTypeData)(T.data)
		if named.underlyingType == nil {
			return nil
		}
		return *(*[]*_interfaceMethodData)(named.underlyingType.data)
	}
	return *(*[]*_interfaceMethodData)(T.data)
}

func interfaceIsAssignable(src *_type, dest *_type) error {
	// Fast path: empty interface (`any`) accepts anything.
	if dest.kind == Interface {
		if len(interfaceMethods(dest)) == 0 {
			return nil
		}
	}

	// For pointer types, the methods live on the pointee's named-type data
	// (this matches what interfaceLookUp does at the dispatch site).
	srcForName := src
	if src.kind == Pointer {
		srcForName = (*_type)(src.data)
	}

	if len(srcForName.name) > 0 && srcForName.kind != Interface && dest.kind == Interface {
		// Concrete named source against an interface destination: the
		// destination interface's methods must all be defined on the
		// source named type.
		srcMethods := ((*_namedTypeData)(srcForName.data)).methods
		destMethods := interfaceMethods(dest)
		for i := range destMethods {
			found := false
			for j := range srcMethods {
				if destMethods[i].id == srcMethods[j].id {
					found = true
					break
				}
			}
			if !found {
				return &TypeAssertError{
					interfaceType: nil,
					concrete:      src,
					asserted:      dest,
					missingMethod: "",
				}
			}
		}
	} else if (src.kind == Interface || (src.kind == Pointer && srcForName.kind == Interface)) && dest.kind == Interface {
		// Interface→interface assertion: every destination method must
		// be present in the source interface's method set.
		srcMethods := interfaceMethods(src)
		if src.kind == Pointer {
			srcMethods = interfaceMethods(srcForName)
		}
		destMethods := interfaceMethods(dest)
		for i := range destMethods {
			found := false
			for j := range srcMethods {
				if destMethods[i].id == srcMethods[j].id {
					found = true
					break
				}
			}
			if !found {
				return &TypeAssertError{
					interfaceType: src,
					concrete:      nil,
					asserted:      dest,
					missingMethod: "",
				}
			}
		}
	} else if src != dest {
		return &TypeAssertError{
			interfaceType: nil,
			concrete:      src,
			asserted:      dest,
			missingMethod: "",
		}
	}
	return nil
}
