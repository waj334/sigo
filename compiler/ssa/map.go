package ssa

import (
	"context"
	"go/ast"
	"go/token"
	"go/types"
	"pkg.si-go.dev/sigo/mlir"
)

func (b *Builder) emitMapRange(ctx context.Context, stmt *ast.RangeStmt) {
	location := b.location(stmt.Pos())
	endLocation := b.location(stmt.End())
	tokLocation := b.location(stmt.TokPos)

	mapType := b.typeOf(ctx, stmt.X).(*types.Map)
	keyT := b.GetStoredType(ctx, mapType.Key())
	elementT := b.GetStoredType(ctx, mapType.Elem())

	// NOTE: The element variable can be omitted from the range statement. There is no need to even consider the element
	//       value if no identifier to hold it is specified.
	keyVar := b.valueOf(ctx, stmt.Key)
	elementVar := b.valueOf(ctx, stmt.Value)

	// Create the exit block where execution will continue following the range statement.
	exitBlock := mlir.BlockCreate2(nil, nil)

	// Create all blocks involved with the for loop.
	rangeBlock := mlir.BlockCreate2(nil, nil)
	appendBlock(ctx, rangeBlock)

	bodyBlock := mlir.BlockCreate2(
		b.types(b.GetStoredType(ctx, mapType.Key()), b.GetStoredType(ctx, mapType.Elem())),
		b.locations(location, location))
	appendBlock(ctx, bodyBlock)

	// Evaluate the map value that will be iterated over.
	X := b.emitExpr(ctx, stmt.X)[0]

	// Branch to the condition block from the current block.
	brOp := mlir.GoCreateBranchOperation(b.ctx, rangeBlock, nil, location)
	appendOperation(ctx, brOp)

	// Build the condition block where the loop condition will continuously be evaluated in.
	buildBlock(ctx, rangeBlock, func() {
		// Create the map range operation.
		op := mlir.GoCreateMapRangeOp(b.ctx, X, bodyBlock, exitBlock, location)
		appendOperation(ctx, op)
	})

	// Build the loop body block.
	buildBlock(ctx, bodyBlock, func() {
		keyValue := mlir.BlockGetArgument(bodyBlock, 0)
		elementValue := mlir.BlockGetArgument(bodyBlock, 1)

		// Any break statement immediately branch to the exit block.
		ctx = newContextWithSuccessorBlock(ctx, exitBlock, nil)

		// Set the predecessor block to the post loop iteration block where any continue statement will branch to.
		ctx = newContextWithPredecessorBlock(ctx, rangeBlock, nil)

		if stmt.Tok == token.DEFINE {
			ident := stmt.Key.(*ast.Ident)
			if identIsValid(ident) {
				// A copy should be emitted into this block. Heap escape analysis should handle converting the stack
				// allocation to a heap allocation in the event that the loop variable escapes the current scope.
				copyAddr := b.emitNamedAlloca(ctx, ident.Name, keyT, b.location(stmt.Key.Pos()))
				keyVar = b.NewTempValue(copyAddr)
				b.setAddr(ctx, ident, keyVar)
			}
		}

		if stmt.Tok == token.DEFINE && stmt.Value != nil {
			ident := stmt.Value.(*ast.Ident)
			if identIsValid(ident) {
				// A copy should be emitted into this block. Heap escape analysis should handle converting the stack
				// allocation to a heap allocation in the event that the loop variable escapes the current scope.
				copyAddr := b.emitNamedAlloca(ctx, ident.Name, elementT, b.location(stmt.Value.Pos()))
				elementVar = b.NewTempValue(copyAddr)
				b.setAddr(ctx, ident, elementVar)
			}
		}

		// Store the loop variable values before executing the loop body.
		if keyVar != nil {
			keyVar.Store(ctx, keyValue, tokLocation)
		}

		if elementVar != nil {
			// Store it at the element variable address.
			elementVar.Store(ctx, elementValue, tokLocation)
		}

		// Emit the loop body
		b.emitBlock(ctx, stmt.Body)

		// NOTE: The loop body can either explicitly terminate or fall off.
		if !blockHasTerminator(currentBlock(ctx)) {
			// Control has fallen off. Branch to the post iteration block.
			brOp := mlir.GoCreateBranchOperation(b.ctx, rangeBlock, nil, endLocation)
			appendOperation(ctx, brOp)
		}
	})

	// Continue emission in the successor block.
	appendBlock(ctx, exitBlock)
	setCurrentBlock(ctx, exitBlock)
}
