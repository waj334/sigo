package errors

type strError string

func (s strError) Error() string {
	return string(s)
}

func New(msg string) error {
	return strError(msg)
}
