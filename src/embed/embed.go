// Package embed provides access to files embedded in the running Go program.
//
// Go source files that import "embed" can use the //go:embed directive to
// initialize a variable of type string or []byte from the contents of files
// read at compile time.
package embed
