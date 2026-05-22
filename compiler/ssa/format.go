package ssa

import (
	"fmt"
	"go/types"
	"hash/fnv"
	"io"
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

// promotedTrampolineSymbol returns the deterministic symbol name for the
// trampoline emitted on `outerNamed` for the promoted method named `methodName`.
// Both createNamedType (when populating the named type's method list) and
// createPromotedMethodTrampoline (when emitting the trampoline body) must
// use this same name.
func promotedTrampolineSymbol(outerNamed *types.Named, methodName string) string {
	sym := fmt.Sprintf("%s_promoted_%s",
		qualifiedName(outerNamed.Obj().Name(), outerNamed.Obj().Pkg()), methodName)
	if outerNamed.TypeArgs().Len() > 0 {
		h := fnv.New64a()
		for i := 0; i < outerNamed.TypeArgs().Len(); i++ {
			io.WriteString(h, outerNamed.TypeArgs().At(i).String())
			h.Write([]byte{0})
		}
		sym += fmt.Sprintf("_%016x", h.Sum64())
	}
	return sym
}
