package ssa

import (
	"context"
	"go/ast"
	"go/token"
	"go/types"

	"pkg.si-go.dev/go-mlir/mlir"
	"pkg.si-go.dev/sigo/goir/binding/goir"
)

func (b *Builder) emitArrayRange(ctx context.Context, stmt *ast.RangeStmt) {
	location := b.location(ctx, stmt.Pos())
	endLocation := b.location(ctx, stmt.End())
	tokLocation := b.location(ctx, stmt.TokPos)

	var arrayType *types.Array
	switch xt := b.typeOf(ctx, stmt.X).(type) {
	case *types.Array:
		arrayType = xt
	case *types.Pointer:
		arrayType = xt.Elem().(*types.Array)
	default:
		panic("emitArrayRange: unexpected type " + xt.String())
	}
	elementT := b.GetStoredType(ctx, arrayType.Elem())
	ptrT := b.GetStoredType(ctx, types.NewPointer(arrayType.Elem()))

	var keyT mlir.TypeLike = b.si
	if keyType := b.typeOf(ctx, stmt.Key); keyType != nil {
		keyT = b.GetStoredType(ctx, keyType)
	}

	// Create the exit block where execution will continue following the range statement.
	exitBlock := mlir.NewBlock(nil, nil)

	// Create all blocks involved with the for loop.
	condBlock := mlir.NewBlock(b.types(b.si), b.locations(b._noLoc))
	appendBlock(ctx, condBlock)

	bodyBlock := mlir.NewBlock(b.types(keyT), b.locations(b._noLoc))
	appendBlock(ctx, bodyBlock)

	postIterBlock := mlir.NewBlock(b.types(keyT), b.locations(b._noLoc))
	appendBlock(ctx, postIterBlock)

	// Evaluate the base address of the array that will be iterated over.
	// For an array value, take its address. For *[N]T, the pointer's value
	// is already the array base address — evaluating the expression yields
	// it directly.
	var X mlir.ValueLike
	if _, isPtr := b.typeOf(ctx, stmt.X).(*types.Pointer); isPtr {
		X = b.emitExpr(ctx, stmt.X)[0]
	} else {
		X = b.addressOf(ctx, stmt.X, location)
	}

	// Create the length value of the array.
	lenValue := b.emitConstInt(ctx, arrayType.Len(), keyT, location)

	// NOTE: The value variable can be omitted from the range statement. There is no need to even consider the element
	//       value if no identifier to hold it is specified.
	keyVar := b.valueOf(ctx, stmt.Key)
	elementVar := b.valueOf(ctx, stmt.Value)

	// Branch to the condition block from the current block.
	zeroValue := b.emitConstInt(ctx, 0, keyT, location)
	brOp := goir.NewBranchOperation(b.ctx, condBlock, b.values(zeroValue), location)
	appendOperation(ctx, brOp)

	// Build the condition block where the loop condition will continuously be evaluated in.
	buildBlock(ctx, condBlock, func() {
		keyValue := condBlock.Argument(0)

		// Compare the iterator value against the array length value.
		cmpOp := goir.NewCmpIOperation(b.ctx, b.i1, b.cmpIPredicate(token.LSS, isUnsigned(keyT)), keyValue, lenValue, location)
		appendOperation(ctx, cmpOp)
		cond := resultOf(cmpOp)

		// Conditionally branch to the loop body block if the loop condition evaluates to true. Otherwise, branch to the
		// exit block.
		condBrOp := goir.NewCondBranchOperation(b.ctx, cond, bodyBlock, b.values(keyValue), exitBlock, nil, location)
		appendOperation(ctx, condBrOp)
	})

	// Build the loop body block.
	buildBlock(ctx, bodyBlock, func() {
		keyValue := bodyBlock.Argument(0)

		// Any break statement immediately branch to the exit block.
		ctx = newContextWithSuccessorBlock(ctx, exitBlock, nil)

		// Set the predecessor block to the post loop iteration block where any continue statement will branch to.
		ctx = newContextWithPredecessorBlock(ctx, postIterBlock, b.values(keyValue))

		if stmt.Tok == token.DEFINE {
			ident := stmt.Key.(*ast.Ident)
			if identIsValid(ident) {
				// A copy should be emitted into this block. Heap escape analysis should handle converting the stack
				// allocation to a heap allocation in the event that the loop variable escapes the current scope.
				copyAddr := b.emitNamedAlloca(ctx, ident.Name, keyT, b.typeOf(ctx, stmt.Key), b.location(ctx, stmt.Key.Pos()))
				keyVar = b.NewTempValue(copyAddr)
				b.setAddr(ctx, ident, keyVar)
			}
		}

		if stmt.Tok == token.DEFINE && stmt.Value != nil {
			ident := stmt.Value.(*ast.Ident)
			if identIsValid(ident) {
				// A copy should be emitted into this block. Heap escape analysis should handle converting the stack
				// allocation to a heap allocation in the event that the loop variable escapes the current scope.
				copyAddr := b.emitNamedAlloca(ctx, ident.Name, elementT, arrayType.Elem(), b.location(ctx, stmt.Value.Pos()))
				elementVar = b.NewTempValue(copyAddr)
				b.setAddr(ctx, ident, elementVar)
			}
		}

		// Store the loop variable values before executing the loop body.
		if keyVar != nil {
			keyVar.Store(ctx, keyValue, tokLocation)
		}

		if elementVar != nil {
			// Get the address of the array element at the current iterator index.
			gepOp := goir.NewGepOperation(b.ctx, X, elementT, nil, []mlir.ValueLike{keyValue}, []bool{true}, ptrT, location)
			appendOperation(ctx, gepOp)

			// Load the value from the array.
			value := b.emitLoad(ctx, resultOf(gepOp), elementT, location)

			// Store it at the element variable address.
			elementVar.Store(ctx, value, tokLocation)
		}

		// Emit the loop body.
		b.emitBlock(ctx, stmt.Body)

		if !blockHasTerminator(currentBlock(ctx)) {
			// Branch to the post iteration block.
			brOp := goir.NewBranchOperation(b.ctx, postIterBlock, b.values(keyValue), endLocation)
			appendOperation(ctx, brOp)
		}
	})

	// Build the post iteration block.
	buildBlock(ctx, postIterBlock, func() {
		keyValue := postIterBlock.Argument(0)

		// Increment the iterator value by one.
		oneValue := b.emitConstInt(ctx, 1, keyT, endLocation)
		addOp := goir.NewAddIOperation(b.ctx, keyT, keyValue, oneValue, endLocation)
		appendOperation(ctx, addOp)

		// branch to the condition block.
		brOp := goir.NewBranchOperation(b.ctx, condBlock, resultsOf(addOp), endLocation)
		appendOperation(ctx, brOp)
	})

	// Continue emission in the successor block.
	appendBlock(ctx, exitBlock)
	setCurrentBlock(ctx, exitBlock)
}
