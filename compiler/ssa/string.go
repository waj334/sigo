package ssa

import (
	"context"
	"go/ast"
	"go/token"
	"pkg.si-go.dev/sigo/mlir"
)

func (b *Builder) emitStringRange(ctx context.Context, stmt *ast.RangeStmt) {
	location := b.location(stmt.Pos())
	endLocation := b.location(stmt.End())
	tokLocation := b.location(stmt.TokPos)

	keyT := b.si
	if keyType := b.typeOf(ctx, stmt.Key); keyType != nil {
		keyT = b.GetStoredType(ctx, keyType)
	}

	// Create the exit block where execution will continue following the range statement.
	exitBlock := mlir.BlockCreate2(nil, nil)

	// Create all blocks involved with the for loop.
	rangeBlock := mlir.BlockCreate2(nil, nil)
	appendBlock(ctx, rangeBlock)

	bodyBlock := mlir.BlockCreate2(b.types(keyT, b.si32), b.locations(location, location))
	appendBlock(ctx, bodyBlock)

	keyVar := b.valueOf(ctx, stmt.Key)
	valueVar := b.valueOf(ctx, stmt.Value)

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

		// Any break statement immediately branch to the exit block.
		ctx = newContextWithSuccessorBlock(ctx, exitBlock, nil)

		// Any continue statement should branch to the condition block.
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
				elementT := b.GetStoredType(ctx, b.typeOf(ctx, stmt.Value))
				copyAddr := b.emitNamedAlloca(ctx, ident.Name, elementT, b.location(stmt.Value.Pos()))
				valueVar = b.NewTempValue(copyAddr)
				b.setAddr(ctx, ident, valueVar)
			}
		}

		// Store the loop variable values before executing the loop body.
		if keyVar != nil {
			keyVar.Store(ctx, keyValue, tokLocation)
		}

		if valueVar != nil {
			valueVar.Store(ctx, elementValue, tokLocation)
		}

		// Emit the loop body
		b.emitBlock(ctx, stmt.Body)

		// NOTE: The loop body can either explicitly terminate or falls off.
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
