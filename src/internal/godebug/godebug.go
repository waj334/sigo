package godebug

type Setting struct{}

func New(name string) *Setting {
	return &Setting{}
}

func (s *Setting) Name() string       { return "" }
func (s *Setting) Undocumented() bool { return false }
func (s *Setting) String() string     { return "" }
func (s *Setting) IncNonDefault()     {}
func (s *Setting) Value() string      { return "" }
