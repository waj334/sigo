package ssa

import (
	"go/types"
)

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
		recvT := signature.Recv().Type()
		if isPointer(recvT) {
			recvT = recvT.(*types.Pointer).Elem()
		}
		if named, ok := recvT.(*types.Named); ok {
			typename = named.Obj().Name()
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
	return symbol
}
