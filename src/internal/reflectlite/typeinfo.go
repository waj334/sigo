package reflectlite

import "unsafe"

type _type struct {
	kind Kind
	size uintptr
	data unsafe.Pointer
	name *string
}

func (t *_type) Name() string {
	if t.kind == Invalid && t.data != nil {
		//namedType := (*_namedTypeData)(t.data)
		return ""
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
	return t.kind
}

func (t *_type) Implements(u Type) bool {
	if u == nil {
		panic("reflect: nil type passed to Type.Implements")
	}
	if u.Kind() != Interface {
		panic("reflect: non-interface type passed to Type.Implements")
	}

	var tmethods []_funcData
	if t.name != nil {
		namedType := (*_namedTypeData)(t.data)
		tmethods = namedType.methods
	} else {
		tmethods = *((*[]_funcData)(t.data))
	}

	uface := (*_interface)(unsafe.Pointer(&u))
	utype := (*_type)(uface.typePtr.data)

	var umethods []_funcData
	if utype.name != nil {
		namedType := (*_namedTypeData)(utype.data)
		tmethods = namedType.methods
	} else {
		tmethods = *((*[]_funcData)(utype.data))
	}

	// TODO: This search is quadratic. The method table can be sorted such that the interface methods will appear in
	//       the same order in both method tables! Refactor this!
	for i := 0; i < len(tmethods); i++ {
		tsignature := (*_signatureTypeData)(tmethods[i].signature.data)
		for j := 0; j < len(umethods); j++ {
			usignature := (*_signatureTypeData)(umethods[j].signature.data)
			// Check if signature hashes do not match
			if usignature.id != tsignature.id {
				// Type T does not implement type U
				return false
			}
		}
	}

	return true
}

func (t *_type) AssignableTo(u Type) bool {
	utype := u.(*_type)
	if t == utype {
		// Identical types
		return true
	}

	return false
}

func (t *_type) Comparable() bool {
	//TODO implement me
	panic("implement me")
}

func (t *_type) String() string {
	// TODO
	return ""
}

func (t *_type) Elem() Type {
	switch t.Kind() {
	case Array:
		tt := (*_arrayTypeData)(t.data)
		return tt.elementType
	case Chan:
		tt := (*_channelTypeData)(t.data)
		return tt.elementType
	case Map:
		tt := (*_mapTypeData)(t.data)
		return tt.valueType
	case Pointer:
		return (*_type)(t.data)
	case Slice:
		return (*_type)(t.data)
	}
	panic("reflect: Elem of invalid type")
}
