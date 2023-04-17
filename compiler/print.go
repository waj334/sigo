package compiler

import "fmt"

func (c *Compiler) println(verbosity Verbosity, args ...any) {
	if c.options.Verbosity >= verbosity {
		fmt.Println(args...)
	}
}

func (c *Compiler) printf(verbosity Verbosity, format string, args ...any) {
	if c.options.Verbosity >= verbosity {
		fmt.Printf(format, args...)
	}
}
