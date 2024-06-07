package runtime

import "unsafe"

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

func interfaceAssert(X _interface, T *_type, hasOk bool) (result _interface, ok bool) {
	err := interfaceIsAssignable(X.valueT, T)
	if err != nil {
		if hasOk {
			return _interface{}, false
		} else {
			panic(err)
		}
	}
	return _interface{
		value:  X.value,
		valueT: T,
	}, true
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
			// TODO
			return false
		case Complex128:
			// TODO
			return false
		case String:
			return *(*string)(X.value) == *(*string)(otherValue)
		case UnsafePointer:
			return *(*unsafe.Pointer)(X.value) == *(*unsafe.Pointer)(otherValue)
		}
	}
	return false
}

func interfaceLookUp(i _interface, id uint32) (receiver, result unsafe.Pointer) {
	T := i.valueT

	// Get the element type of the pointer
	if i.valueT.kind == Pointer {
		T = (*_type)(i.valueT.data)
	}

	// Locate the method matching the id
	elementType := (*_namedTypeData)(T.data)
	for _, method := range elementType.methods {
		methodId := method.id
		if id == methodId {
			signature := (*_signatureTypeData)(method.signature.data)
			receiver = i.value
			if i.valueT.kind == Pointer {
				// Load the receiver pointer value.
				receiver = *(*unsafe.Pointer)(receiver)
				if signature.receiverType.kind != Pointer {
					// Load the value.
					receiver = *(*unsafe.Pointer)(receiver)
				}
			} else {
				if signature.receiverType.kind == Pointer {
					// Pass the address of the underlying value.
					receiver = *(*unsafe.Pointer)(&i.value)
				} // else the receiver value is already holds the address of the value.
			}
			return receiver, method.funcPtr
		}
	}
	panic("no concrete implementation found")
}

func interfaceIsAssignable(src *_type, dest *_type) error {
	if len(src.name) > 0 && dest.kind == Interface {
		// The destination methods must be defined on the source named type.
		srcMethods := ((*_namedTypeData)(src.data)).methods
		destMethods := *(*[]_interfaceMethodData)(dest.data)
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
	} else if src.kind == Interface && dest.kind == Interface {
		// The destination type's methods must be defined on the source interface type.
		// The source and destination types must be the same.
		srcMethods := *(*[]_interfaceMethodData)(src.data)
		destMethods := *(*[]_interfaceMethodData)(dest.data)
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
