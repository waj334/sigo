package check

import (
	"go/types"
	"strings"
)

func qualifiedName(name string, p *types.Package) string {
	if p != nil {
		name = p.Path() + "." + name
	}
	return name
}

func cleanStr(input string) string {
	input = strings.Trim(input, "`\"")
	input = strings.TrimSpace(input)
	return input
}

func typeIsPointer(T types.Type) bool {
	switch T := types.Unalias(T).(type) {
	case *types.Basic:
		return T.Kind() == types.UnsafePointer
	case *types.Pointer:
		return true
	default:
		return false
	}
}
