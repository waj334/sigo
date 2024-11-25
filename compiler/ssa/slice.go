package ssa

import (
	"context"
	"go/ast"
	"go/token"
	"go/types"
	"omibyte.io/sigo/mlir"
)

func (b *Builder) emitSliceRange(ctx context.Context, stmt *ast.RangeStmt) {
	location := b.location(stmt.Pos())
	endLocation := b.location(stmt.End())
	tokLocation := b.location(stmt.TokPos)

	sliceType := b.typeOf(ctx, stmt.X).(*types.Slice)
	elementT := b.GetStoredType(ctx, sliceType.Elem())
	ptrT := mlir.GoCreatePointerType(elementT)

	keyT := b.si
	if keyType := b.typeOf(ctx, stmt.Key); keyType != nil {
		keyT = b.GetStoredType(ctx, keyType)
	}

	// Create the exit block where execution will continue following the range statement.
	exitBlock := mlir.BlockCreate2(nil, nil)

	// Create all blocks involved with the for loop.
	condBlock := mlir.BlockCreate2(b.types(keyT), b.locations(b._noLoc))
	appendBlock(ctx, condBlock)

	bodyBlock := mlir.BlockCreate2(b.types(keyT), b.locations(b._noLoc))
	appendBlock(ctx, bodyBlock)

	postIterBlock := mlir.BlockCreate2(b.types(keyT), b.locations(b._noLoc))
	appendBlock(ctx, postIterBlock)

	// Evaluate the slice value that will be iterated over.
	X := b.emitExpr(ctx, stmt.X)[0]

	// Get the underlying array of the slice
	sliceDataCallOp := mlir.GoCreateBuiltInCallOperation(b.ctx, "unsafe.SliceData", b.types(b.ptr), b.values(X), location)
	appendOperation(ctx, sliceDataCallOp)
	arrValue := resultOf(sliceDataCallOp)

	// Get the length of the slice.
	lenCallOp := mlir.GoCreateBuiltInCallOperation(b.ctx, "len", b.types(b.si), b.values(X), location)
	appendOperation(ctx, lenCallOp)
	lenValue := resultOf(lenCallOp)

	// NOTE: The element variable can be omitted from the range statement. There is no need to even consider the element
	//       value if no identifier to hold it is specified.
	keyVar := b.valueOf(ctx, stmt.Key)
	elementVar := b.valueOf(ctx, stmt.Value)

	// Branch to the condition block from the current block.
	zeroValue := b.emitConstInt(ctx, 0, keyT, location)
	brOp := mlir.GoCreateBranchOperation(b.ctx, condBlock, b.values(zeroValue), location)
	appendOperation(ctx, brOp)

	// Build the condition block where the loop condition will continuously be evaluated in.
	buildBlock(ctx, condBlock, func() {
		keyValue := mlir.BlockGetArgument(condBlock, 0)

		// Compare the iterator value against the array length value.
		cmpOp := mlir.GoCreateCmpIOperation(b.ctx, b.i1, b.cmpIPredicate(token.LSS, isUnsigned(keyT)), keyValue, lenValue, location)
		appendOperation(ctx, cmpOp)
		cond := resultOf(cmpOp)

		// Conditionally branch to the loop body block if the loop condition evaluates to true. Otherwise, branch to the
		// exit block.
		condBrOp := mlir.GoCreateCondBranchOperation(b.ctx, cond, bodyBlock, b.values(keyValue), exitBlock, nil, location)
		appendOperation(ctx, condBrOp)
	})

	// Build the loop body block.
	buildBlock(ctx, bodyBlock, func() {
		keyValue := mlir.BlockGetArgument(bodyBlock, 0)

		// Any break statement immediately branch to the exit block.
		ctx = newContextWithSuccessorBlock(ctx, exitBlock, nil)

		// Set the predecessor block to the post loop iteration block where any continue statement will branch to.
		ctx = newContextWithPredecessorBlock(ctx, postIterBlock, b.values(keyValue))

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
			// Get the address of the array element at the current iterator index.
			gepOp := mlir.GoCreateGepOperation2(b.ctx, arrValue, elementT, []any{keyValue}, ptrT, location)
			appendOperation(ctx, gepOp)

			// Load the value from the array.
			loadOp := mlir.GoCreateLoadOperation(b.ctx, resultOf(gepOp), elementT, location)
			appendOperation(ctx, loadOp)

			// Store it at the element variable address.
			elementVar.Store(ctx, resultOf(loadOp), tokLocation)
		}

		// Emit the loop body.
		b.emitBlock(ctx, stmt.Body)

		if !blockHasTerminator(currentBlock(ctx)) {
			// Branch to the post iteration block.
			brOp := mlir.GoCreateBranchOperation(b.ctx, postIterBlock, b.values(keyValue), endLocation)
			appendOperation(ctx, brOp)
		}
	})

	// Build the post iteration block.
	buildBlock(ctx, postIterBlock, func() {
		keyValue := mlir.BlockGetArgument(postIterBlock, 0)

		// Increment the iterator value by one.
		oneValue := b.emitConstInt(ctx, 1, keyT, endLocation)
		addOp := mlir.GoCreateAddIOperation(b.ctx, keyT, keyValue, oneValue, endLocation)
		appendOperation(ctx, addOp)

		// branch to the condition block.
		brOp := mlir.GoCreateBranchOperation(b.ctx, condBlock, b.values(resultOf(addOp)), endLocation)
		appendOperation(ctx, brOp)
	})

	// Continue emission in the successor block.
	appendBlock(ctx, exitBlock)
	setCurrentBlock(ctx, exitBlock)
}
