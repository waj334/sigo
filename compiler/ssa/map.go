package ssa

import (
	"context"
	"go/ast"
	"go/types"
	"omibyte.io/sigo/mlir"
)

func (b *Builder) emitMapRange(ctx context.Context, stmt *ast.RangeStmt) {
	location := b.location(stmt.Pos())

	mapType := b.typeOf(ctx, stmt.X).(*types.Map)

	keyVar := b.valueOf(ctx, stmt.Key)
	if keyVar == nil {
		obj := b.objectOf(ctx, stmt.Key)
		keyT := b.GetStoredType(ctx, obj.Type())
		keyVar = b.emitLocalVar(ctx, obj, keyT, false)
	}

	var elementVar Value
	if stmt.Value != nil {
		elementVar = b.valueOf(ctx, stmt.Value)
		if elementVar == nil {
			obj := b.objectOf(ctx, stmt.Value)
			elementT := b.GetStoredType(ctx, obj.Type())
			elementVar = b.emitLocalVar(ctx, obj, elementT, false)
		}
	}

	// Create the exit block where execution will continue following the range statement.
	exitBlock := mlir.BlockCreate2(nil, nil)

	// Create all blocks involved with the for loop.
	rangeBlock := mlir.BlockCreate2(nil, nil)
	appendBlock(ctx, rangeBlock)

	bodyBlock := mlir.BlockCreate2(
		b.types(b.GetStoredType(ctx, mapType.Key()), b.GetStoredType(ctx, mapType.Elem())),
		b.locations(location, location))
	appendBlock(ctx, bodyBlock)

	// Any break statement immediately branch to the exit block.
	ctx = newContextWithSuccessorBlock(ctx, exitBlock)

	// Any continue statement should branch to the condition block.
	ctx = newContextWithPredecessorBlock(ctx, rangeBlock)

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

		// Set the predecessor block to the post loop iteration block where any continue statement will branch to.
		ctx = newContextWithPredecessorBlock(ctx, rangeBlock)

		// TODO: Create new loop variables for these.
		// Update the loop variables.
		keyVar.Store(ctx, keyValue, location)
		if elementVar != nil {
			elementVar.Store(ctx, elementValue, location)
		}

		// Emit the loop body
		b.emitBlock(ctx, stmt.Body)

		// NOTE: The loop body can either explicitly terminate or falls off.

		if !blockHasTerminator(currentBlock(ctx)) {
			// Control has fallen off. Branch to the post iteration block.
			brOp := mlir.GoCreateBranchOperation(b.ctx, rangeBlock, nil, location)
			appendOperation(ctx, brOp)
		}
	})

	// Continue emission in the successor block.
	appendBlock(ctx, exitBlock)
	setCurrentBlock(ctx, exitBlock)
}
