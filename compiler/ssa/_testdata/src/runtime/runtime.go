package runtime

import "unsafe"

// Mocks for test code generator.

type (
	_channel   struct{}
	_interface struct {
		ptr unsafe.Pointer
		T   *_type
	}
	_map struct{}

	_mapIterator struct {
		m      _map
		bucket int
		entry  *struct{}
	}
	_slice struct {
		arr unsafe.Pointer
		l   int
		c   int
	}
	_string struct {
		arr unsafe.Pointer
		l   int
	}

	_stringIterator struct {
		str   _string
		index int
	}

	_type                struct{}
	_namedTypeData       struct{}
	_funcData            struct{}
	_interfaceMethodData struct{}
	_signatureTypeData   struct{}
	_arrayTypeData       struct{}
	_structTypeData      struct{}
	_structFieldData     struct{}
	_channelTypeData     struct{}
	_mapTypeData         struct{}
)

func deferStartStack() {}
func deferRun()        {}
