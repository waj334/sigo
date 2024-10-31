package ssa

import (
	"go/types"
)

func MangleSymbol(symbol string) string {
	return mangleSymbol(symbol)
}

func mangleSymbol(symbol string) string {
	return symbol
}

func demangleSymbol(symbol string) string {
	return symbol
}

/*
// NOTE: Name mangling is disabled for now.
func mangleSymbol(symbol string) string {
	symbol = strings.ReplaceAll(symbol, "/", "$")
	symbol = strings.ReplaceAll(symbol, ".", "@")
	return symbol
}

func demangleSymbol(symbol string) string {
	symbol = strings.ReplaceAll(symbol, "$", "/")
	symbol = strings.ReplaceAll(symbol, "@", ".")
	return symbol
}
*/

func qualifiedName(name string, p *types.Package) string {
	if p != nil {
		name = p.Path() + "." + name
	}
	return name
}

func qualifiedName2(path, name string) string {
	name = path + "." + name
	return name
}

func qualifiedFuncName(obj *types.Func) string {
	signature := obj.Type().(*types.Signature)

	// Get the name of the method receiver's named type.
	var typename string
	if signature.Recv() != nil {
		if isPointer(signature.Recv().Type()) {
			typename = signature.Recv().Type().(*types.Pointer).Elem().(*types.Named).Obj().Name()
		} else {
			typename = signature.Recv().Type().(*types.Named).Obj().Name()
		}
	}

	// Format the callee.
	if len(typename) > 0 {
		return qualifiedName(typename+"."+obj.Name(), obj.Pkg())
	}
	return qualifiedName(obj.Name(), obj.Pkg())
}

func (b *Builder) resolveSymbol(symbol string) string {
	symbolInfo := b.config.Program.Symbols.GetSymbolInfo(symbol)
	if len(symbolInfo.LinkName) > 0 {
		symbol = symbolInfo.LinkName
	}
	return mangleSymbol(symbol)
}
