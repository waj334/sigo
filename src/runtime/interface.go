package runtime

import "unsafe"

type interfaceDescriptor struct {
	typePtr  *typeDescriptor
	valuePtr unsafe.Pointer
}

//go:export interfaceMake runtime.makeInterface
func interfaceMake(value unsafe.Pointer, valueType *typeDescriptor) interfaceDescriptor {
	return interfaceDescriptor{
		typePtr:  valueType,
		valuePtr: value,
	}
}

//go:export interfaceAssert runtime.typeAssert
func interfaceAssert(X *interfaceDescriptor, from *typeDescriptor, T *typeDescriptor, hasOk bool) (unsafe.Pointer, bool) {
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

//go:export interfaceCompare runtime.interfaceCompare
func interfaceCompare(X unsafe.Pointer, Y unsafe.Pointer) bool {
	xi := (*interfaceDescriptor)(X)
	yi := (*interfaceDescriptor)(Y)

	// Nil comparison
	if xi.typePtr == yi.typePtr && xi.valuePtr == nil && yi.valuePtr == nil {
		return true
	}

	// Interfaces are equal if their types are the same and their values are the same
	if xi.typePtr == yi.typePtr {
		switch xi.typePtr.kind {
		case Bool:
			return *(*bool)(xi.valuePtr) == *(*bool)(yi.valuePtr)
		case Int:
			return *(*int)(xi.valuePtr) == *(*int)(yi.valuePtr)
		case Int8:
			return *(*int8)(xi.valuePtr) == *(*int8)(yi.valuePtr)
		case Int16:
			return *(*int16)(xi.valuePtr) == *(*int16)(yi.valuePtr)
		case Int32:
			return *(*int32)(xi.valuePtr) == *(*int32)(yi.valuePtr)
		case Int64:
			return *(*int64)(xi.valuePtr) == *(*int64)(yi.valuePtr)
		case Uint:
			return *(*uint)(xi.valuePtr) == *(*uint)(yi.valuePtr)
		case Uint8:
			return *(*uint8)(xi.valuePtr) == *(*uint8)(yi.valuePtr)
		case Uint16:
			return *(*uint16)(xi.valuePtr) == *(*uint16)(yi.valuePtr)
		case Uint32:
			return *(*uint32)(xi.valuePtr) == *(*uint32)(yi.valuePtr)
		case Uint64:
			return *(*uint64)(xi.valuePtr) == *(*uint64)(yi.valuePtr)
		case Uintptr:
			return *(*uintptr)(xi.valuePtr) == *(*uintptr)(yi.valuePtr)
		case Float32:
			return *(*float32)(xi.valuePtr) == *(*float32)(yi.valuePtr)
		case Float64:
			return *(*float64)(xi.valuePtr) == *(*float64)(yi.valuePtr)
		case Complex64:
			// TODO
			return false
		case Complex128:
			// TODO
			return false
		case String:
			return *(*string)(xi.valuePtr) == *(*string)(yi.valuePtr)
		case UnsafePointer:
			return *(*unsafe.Pointer)(xi.valuePtr) == *(*unsafe.Pointer)(yi.valuePtr)
		}
	}
	return false
}

func interfaceLookUp(ptr unsafe.Pointer, id uint32) (result unsafe.Pointer) {
	iface := (*interfaceDescriptor)(ptr)
	info := iface.typePtr

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
