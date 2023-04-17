package compiler

type Verbosity int

const (
	Quiet Verbosity = iota
	Info
	Warning
	Debug
)

type Options struct {
	Target            *Target
	LinkNames         map[string]string
	GenerateDebugInfo bool
	Verbosity         Verbosity
}
