package runtime

type Error interface {
	error
	RuntimeError()
}

type TypeAssertError struct {
	_interface    *_type
	concrete      *_type
	asserted      *_type
	missingMethod string
}

func (*TypeAssertError) RuntimeError() {}

func (e *TypeAssertError) Error() string {
	inter := "interface"
	if e._interface != nil {
		inter = *e._interface.name
	}
	as := *e.asserted.name
	if e.concrete == nil {
		return "interface conversion: " + inter + " is nil, not " + as
	}
	cs := *e.concrete.name
	if len(e.missingMethod) == 0 {
		msg := "interface conversion: " + inter + " is " + cs + ", not " + as
		// TODO: Need to store package path in type information
		/*if cs == as {
			e.concrete.pkgpath() != e.asserted.pkgpath() {
				msg += " (types from different packages)"
			} else {
				msg += " (types from different scopes)"
			}
		}*/
		return msg
	}

	return "interface conversion: " + cs + " is not " +
		as + ": missing method " + e.missingMethod
}
