package ssa

import (
	"context"
	"go/ast"
	"go/token"
	"go/types"
	"strings"

	"pkg.si-go.dev/go-mlir/mlir"
)

// appendOperation Appends the operation to the last block in the region provided by the context.
func appendOperation(ctx context.Context, op mlir.Operation) {
	block := currentBlock(ctx)
	block.AppendOwnedOperation(op)
}

func appendBlock(ctx context.Context, block mlir.Block) {
	currentRegion(ctx).AppendOwnedBlock(block)
}

func (b *Builder) appendToModule(operation mlir.Operation) {
	b.config.Module.Body().AppendOwnedOperation(operation)
}

func resultOf(op mlir.Operation) mlir.Result {
	return op.Result(0)
}

func resultsOf(op mlir.Operation) []mlir.ValueLike {
	var results []mlir.ValueLike
	for i := 0; i < op.NumResults(); i++ {
		results = append(results, op.Result(i))
	}
	return results
}

func blockHasTerminator(block mlir.Block) bool {
	return !block.Terminator().IsNull()
}

func buildBlock(ctx context.Context, block mlir.Block, fn func()) {
	// Save the current block.
	_block := currentBlock(ctx)

	// Switch the current block to the input block.
	setCurrentBlock(ctx, block)

	// Fill the block.
	fn()

	// Reset the current block back to the previous block.
	setCurrentBlock(ctx, _block)
}

func fill[T any](s []T, v T) []T {
	for i := range s {
		s[i] = v
	}
	return s
}

func isPredeclaration(decl *ast.FuncDecl) bool {
	if decl == nil {
		return false
	}
	return decl.Body == nil || decl.Body.Lbrace == token.NoPos || decl.Body.Rbrace == token.NoPos
}

func tupleTypes(tuple *types.Tuple) []types.Type {
	result := make([]types.Type, tuple.Len())
	for i := 0; i < tuple.Len(); i++ {
		result[i] = tuple.At(i).Type()
	}
	return result
}

func (b *Builder) namedOf(name string, attr mlir.AttributeLike) mlir.NamedAttribute {
	return mlir.NewNamedAttribute(name, attr)
}

func identIsValid(ident *ast.Ident) bool {
	return len(ident.Name) > 0 && ident.Name != "_"
}

func cleanConstString(input string) string {
	result := strings.ReplaceAll(input, "\"", "")
	result = strings.ReplaceAll(result, "`", "")
	return result
}
