package ssa

import (
	"context"
	"go/ast"
	"go/token"
	"go/types"

	"omibyte.io/sigo/mlir"
)

// appendOperation Appends the operation to the last block in the region provided by the context.
func appendOperation(ctx context.Context, op mlir.Operation) {
	block := currentBlock(ctx)
	mlir.BlockAppendOwnedOperation(block, op)
}

func appendBlock(ctx context.Context, block mlir.Block) {
	mlir.RegionAppendOwnedBlock(currentRegion(ctx), block)
}

func (b *Builder) appendToModule(operation mlir.Operation) {
	moduleBlock := mlir.ModuleGetBody(b.config.Module)
	mlir.BlockAppendOwnedOperation(moduleBlock, operation)
}

func resultOf(op mlir.Operation) mlir.Value {
	return mlir.OperationGetResult(op, 0)
}

func resultsOf(op mlir.Operation) []mlir.Value {
	var results []mlir.Value
	for i := 0; i < mlir.OperationGetNumResults(op); i++ {
		results = append(results, mlir.OperationGetResult(op, i))
	}
	return results
}

func blockHasTerminator(block mlir.Block) bool {
	return !mlir.OperationIsNull(mlir.BlockGetTerminator(block))
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

func (b *Builder) namedOf(name string, attr mlir.Attribute) mlir.NamedAttribute {
	return mlir.NamedAttributeGet(mlir.IdentifierGet(b.ctx, name), attr)
}
