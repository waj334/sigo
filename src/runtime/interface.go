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

func interfaceAssert(X *_interface, from *_type, T *_type, hasOk bool) (result unsafe.Pointer, ok bool) {
	var err error
	if X == nil {
		if T.ptr == nil {
			return nil, true
		} else {
			err = &TypeAssertError{
				_interface:    from,
				concrete:      nil,
				asserted:      T,
				missingMethod: "",
			}
		}
	} else {
		XType := X.typePtr
		if X.typePtr.construct == Pointer {
			XType = X.typePtr.ptr.elementType
		}

		if T.construct != Interface {
			if XType != T { // Underlying test
				err = &TypeAssertError{
					_interface:    from,
					concrete:      XType,
					asserted:      T,
					missingMethod: "",
				}
			} else {
				// Return the concrete value
				return X.valuePtr, true
			}
		} else {
			if T.methods != nil { // Assignable test
				for i := 0; i < T.methods.count; i++ {
					methodNameT := *T.methods.index(i).name
					found := false
					for ii := 0; ii < XType.methods.count; ii++ {
						methodNameX := *XType.methods.index(ii).name
						if methodNameX == methodNameT {
							found = true
							break
						}
					}

					if !found {
						err = &TypeAssertError{
							_interface:    from,
							concrete:      XType,
							asserted:      T,
							missingMethod: methodNameT,
						}
						break
					} else {
						// return a copy of the input interface
						return unsafe.Pointer(&_interface{
							valuePtr: X.valuePtr,
							typePtr:  X.typePtr,
						}), true
					}
				}
			} // else any value can be assigned to this interface
		}
	}

	// Handle error
	if err != nil {
		if !hasOk {
			panic(err)
		}
	}

	return nil, false
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

	// Get the element type of the pointer
	if i.typePtr.ptr != nil {
		info = i.typePtr.ptr.elementType
	}

	// Locate the method matching the id
	for method := 0; method < info.methods.count; method++ {
		fn := info.methods.index(method)
		if id == fn.id {
			return fn.ptr
		}
	}
	panic("no concrete implementation found")
}
