package fmt

// Errorf formats according to a format specifier and returns the string as a
// value that satisfies the error interface.
//
// If the format specifier includes a %w verb with an error operand, the
// returned error will implement an Unwrap method returning the operand. If
// there is more than one %w verb, the returned error will implement an Unwrap
// method returning a []error containing all the %w operands in the order they
// appear in the arguments.
func Errorf(format string, a ...any) error {
	msg := Sprintf(format, a...)

	// Scan for %w verbs to collect wrapped errors.
	var wrapped []error
	argIdx := 0
	i := 0
	for i < len(format) {
		if format[i] != '%' {
			i++
			continue
		}
		i++
		if i >= len(format) {
			break
		}
		if format[i] == '%' {
			i++
			continue
		}
		// Skip flags, width, precision.
		for i < len(format) {
			c := format[i]
			if c == '-' || c == '+' || c == ' ' || c == '0' || c == '#' {
				i++
			} else {
				break
			}
		}
		for i < len(format) && format[i] >= '0' && format[i] <= '9' {
			i++
		}
		if i < len(format) && format[i] == '.' {
			i++
			for i < len(format) && format[i] >= '0' && format[i] <= '9' {
				i++
			}
		}
		if i >= len(format) {
			break
		}
		verb := format[i]
		i++
		if verb == 'w' && argIdx < len(a) {
			if err, ok := a[argIdx].(error); ok {
				wrapped = append(wrapped, err)
			}
		}
		if verb != '%' {
			argIdx++
		}
	}

	switch len(wrapped) {
	case 0:
		return &errorString{msg}
	case 1:
		return &wrapError{msg, wrapped[0]}
	default:
		return &wrapErrors{msg, wrapped}
	}
}

// errorString is a simple error with no wrapped cause.
type errorString struct {
	s string
}

func (e *errorString) Error() string { return e.s }

// wrapError is an error that wraps a single cause.
type wrapError struct {
	msg   string
	cause error
}

func (e *wrapError) Error() string { return e.msg }
func (e *wrapError) Unwrap() error { return e.cause }

// wrapErrors is an error that wraps multiple causes.
type wrapErrors struct {
	msg    string
	causes []error
}

func (e *wrapErrors) Error() string   { return e.msg }
func (e *wrapErrors) Unwrap() []error { return e.causes }
