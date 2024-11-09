package ssa

import (
	"context"
	"go/ast"
	"omibyte.io/sigo/mlir"
)

func (b *Builder) emitStringRange(ctx context.Context, stmt *ast.RangeStmt) {
	location := b.location(stmt.Pos())

	// Create the exit block where execution will continue following the range statement.
	exitBlock := mlir.BlockCreate2(nil, nil)

	// Create all blocks involved with the for loop.
	rangeBlock := mlir.BlockCreate2(nil, nil)
	appendBlock(ctx, rangeBlock)

	bodyBlock := mlir.BlockCreate2(b.types(b.si, b.si32), b.locations(location, location))
	appendBlock(ctx, bodyBlock)

	// Any break statement immediately branch to the exit block.
	ctx = newContextWithSuccessorBlock(ctx, exitBlock)

	// Any continue statement should branch to the condition block.
	ctx = newContextWithPredecessorBlock(ctx, rangeBlock)

	// The key variable is either a new one or an existing one.
	keyVar := b.valueOf(ctx, stmt.Key)
	if keyVar == nil {
		obj := b.objectOf(ctx, stmt.Key)
		keyT := b.GetStoredType(ctx, obj.Type())
		keyVar = b.emitLocalVar(ctx, obj, keyT, false)
	}

	// The value variable is either a new one or an existing one.
	// NOTE: The value variable can be omitted from the range statement.
	var valueVar Value
	if stmt.Value != nil {
		valueVar = b.valueOf(ctx, stmt.Value)
		if valueVar == nil {
			obj := b.objectOf(ctx, stmt.Value)
			elementT := b.GetStoredType(ctx, obj.Type())
			valueVar = b.emitLocalVar(ctx, obj, elementT, false)
		}
	}

	// Evaluate the string value that will be iterated over.
	X := b.emitExpr(ctx, stmt.X)[0]

	// Branch to the condition block from the current block.
	brOp := mlir.GoCreateBranchOperation(b.ctx, rangeBlock, nil, location)
	appendOperation(ctx, brOp)

	// Build the condition block where the loop condition will continuously be evaluated in.
	buildBlock(ctx, rangeBlock, func() {
		op := mlir.GoCreateStringRangeOp(b.ctx, X, bodyBlock, exitBlock, location)
		appendOperation(ctx, op)
	})

	// Build the loop body block.
	buildBlock(ctx, bodyBlock, func() {
		keyValue := mlir.BlockGetArgument(bodyBlock, 0)
		elementValue := mlir.BlockGetArgument(bodyBlock, 1)

		// Set the predecessor block to the post loop iteration block where any continue statement will branch to.
		ctx = newContextWithPredecessorBlock(ctx, rangeBlock)

		// TODO: Create a new loop argument for this.
		if keyVar != nil {
			keyVar.Store(ctx, keyValue, location)
		}

		if valueVar != nil {
			valueVar.Store(ctx, elementValue, location)
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
