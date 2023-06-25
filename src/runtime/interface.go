package runtime

import "unsafe"

type _interface struct {
	typePtr  *_type
	valuePtr unsafe.Pointer
}

func interfaceMake(value unsafe.Pointer, valueType *_type) _interface {
	return _interface{
		typePtr:  valueType,
		valuePtr: value,
	}
}

func interfaceAssert(X *_interface, from *_type, T *_type, hasOk bool) (unsafe.Pointer, bool) {
	var err error
	if X == nil {
		err = &TypeAssertError{
			_interface:    from,
			concrete:      nil,
			asserted:      T,
			missingMethod: "",
		}
	} else if T.construct != Interface && X.typePtr != T {
		err = &TypeAssertError{
			_interface:    from,
			concrete:      X.typePtr,
			asserted:      T,
			missingMethod: "",
		}
	} else if T.construct == Interface && (X.typePtr.construct == Struct || X.typePtr.construct == Interface) {
		for i := 0; i < T.methods.count; i++ {
			methodNameT := *T.methods.index(i).name
			found := false
			for ii := 0; ii < X.typePtr.methods.count; ii++ {
				methodNameX := *X.typePtr.methods.index(ii).name
				if methodNameX == methodNameT {
					found = true
					break
				}
			}

			if !found {
				err = &TypeAssertError{
					_interface:    from,
					concrete:      X.typePtr,
					asserted:      T,
					missingMethod: methodNameT,
				}
				break
			}
		}

		// Return the input interface
		return unsafe.Pointer(X), true
	}

	if err != nil {
		if !hasOk {
			panic(err)
		}
		return nil, false
	}

	return X.valuePtr, true
}

func interfaceCompare(X _interface, Y _interface) bool {
	// Nil comparison
	if X.typePtr == Y.typePtr && X.valuePtr == nil && Y.valuePtr == nil {
		return true
	}

	// Interfaces are equal if their types are the same and their values are the same
	if X.typePtr == Y.typePtr {
		switch X.typePtr.kind {
		case Bool:
			return *(*bool)(X.valuePtr) == *(*bool)(Y.valuePtr)
		case Int:
			return *(*int)(X.valuePtr) == *(*int)(Y.valuePtr)
		case Int8:
			return *(*int8)(X.valuePtr) == *(*int8)(Y.valuePtr)
		case Int16:
			return *(*int16)(X.valuePtr) == *(*int16)(Y.valuePtr)
		case Int32:
			return *(*int32)(X.valuePtr) == *(*int32)(Y.valuePtr)
		case Int64:
			return *(*int64)(X.valuePtr) == *(*int64)(Y.valuePtr)
		case Uint:
			return *(*uint)(X.valuePtr) == *(*uint)(Y.valuePtr)
		case Uint8:
			return *(*uint8)(X.valuePtr) == *(*uint8)(Y.valuePtr)
		case Uint16:
			return *(*uint16)(X.valuePtr) == *(*uint16)(Y.valuePtr)
		case Uint32:
			return *(*uint32)(X.valuePtr) == *(*uint32)(Y.valuePtr)
		case Uint64:
			return *(*uint64)(X.valuePtr) == *(*uint64)(Y.valuePtr)
		case Uintptr:
			return *(*uintptr)(X.valuePtr) == *(*uintptr)(Y.valuePtr)
		case Float32:
			return *(*float32)(X.valuePtr) == *(*float32)(Y.valuePtr)
		case Float64:
			return *(*float64)(X.valuePtr) == *(*float64)(Y.valuePtr)
		case Complex64:
			// TODO
			return false
		case Complex128:
			// TODO
			return false
		case String:
			return *(*string)(X.valuePtr) == *(*string)(Y.valuePtr)
		case UnsafePointer:
			return *(*unsafe.Pointer)(X.valuePtr) == *(*unsafe.Pointer)(Y.valuePtr)
		}
	}
	return false
}

func interfaceLookUp(i _interface, id uint32) (result unsafe.Pointer) {
	info := i.typePtr

	// Get the underlying type
	if info.ptr != nil {
		ptrInfo := info.ptr
		info = ptrInfo.elementType
	}

	for method := 0; method < info.methods.count; method++ {
		fn := info.methods.index(method)
		if id == fn.id {
			return fn.ptr
		}
	}
	panic("no concrete implementation found")
}
