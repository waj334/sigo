package builder

import "omibyte.io/sigo/compiler"

type Options struct {
	Packages          []string
	Output            string
	BuildDir          string
	DumpOnVerifyError bool
	DumpIR            bool
	Environment       Env
	CompilerVerbosity compiler.Verbosity
	GenerateDebugInfo bool
	BuildTags         []string
	CTypeNames        bool
	NumJobs           int
	Optimization      string
	StackSize         int
}
