package runtime

type Error interface {
	error
	RuntimeError()
}

type TypeAssertError struct {
	interfaceType *_type
	concrete      *_type
	asserted      *_type
	missingMethod string
}

func (*TypeAssertError) RuntimeError() {}

func (e *TypeAssertError) Error() string {
	inter := "interface"
	if e.interfaceType != nil {
		//inter = typeName(e.interfaceType)
	}
	//as := typeName(e.asserted)
	as := ""
	if e.concrete == nil {
		return "interface conversion: " + inter + " is nil, not " + as
	}
	//cs := typeName(e.concrete)
	cs := ""
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

// plainError represents a runtime error described a string without
// the prefix "runtime error: " after invoking errorString.Error().
// See Issue #14965.
type plainError string

func (e plainError) RuntimeError() {}

func (e plainError) Error() string {
	return string(e)
}
