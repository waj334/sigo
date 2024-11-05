package ssa

import (
	"context"
	"go/token"

	"go/ast"
	"go/types"

	"omibyte.io/sigo/mlir"
)

func (b *Builder) emitSelectStatement(ctx context.Context, stmt *ast.SelectStmt) {
	bodyBlocks := make([]mlir.Block, 0, len(stmt.Body.List))
	chans := make([]mlir.Value, 0, len(stmt.Body.List))
	isSend := make([]int, 0, len(stmt.Body.List))

	var defaultBlock mlir.Block
	hasDefault := false

	// Create the successor block for this statement.
	successor := mlir.BlockCreate2(nil, nil)

	// Break statements should branch to this statement's immediate successor.
	ctx = newContextWithSuccessorBlock(ctx, successor)

	// Create the clause blocks.
	for _, clause := range stmt.Body.List {
		clause := clause.(*ast.CommClause)
		var block mlir.Block
		if clause.Comm == nil {
			hasDefault = true
			defaultBlock = mlir.BlockCreate2(nil, nil)
			block = defaultBlock
		} else {
			// Extract the specific channel involved in the case clause.
			var value mlir.Value
			send := 0
			switch stmt := clause.Comm.(type) {
			case *ast.AssignStmt:
				value = b.emitExpr(ctx, ast.Unparen(stmt.Rhs[0]).(*ast.UnaryExpr).X.(*ast.Ident))[0]
			case *ast.ExprStmt:
				value = b.emitExpr(ctx, ast.Unparen(stmt.X).(*ast.UnaryExpr).X.(*ast.Ident))[0]
			case *ast.SendStmt:
				value = b.emitExpr(ctx, ast.Unparen(stmt.Chan).(*ast.Ident))[0]
				send = 1
			}

			chans = append(chans, value)
			isSend = append(isSend, send)
			block = mlir.BlockCreate2(nil, nil)
			bodyBlocks = append(bodyBlocks, block)
		}

		// Create the body block
		buildBlock(ctx, block, func() {
			if clause.Comm != nil {
				// Emit the clause statement.
				b.emitStmt(ctx, clause.Comm)
			}

			// Emit the body block statements.
			for _, stmt := range clause.Body {
				b.emitStmt(ctx, stmt)
			}

			if !blockHasTerminator(currentBlock(ctx)) {
				// Branch to the successor block.
				brOp := mlir.GoCreateBranchOperation(b.ctx, successor, nil, b.location(clause.End()))
				appendOperation(ctx, brOp)
			}
		})
		appendBlock(ctx, block)
	}

	sendArr := mlir.DenseBoolArrayGet(b.ctx, isSend)
	if !hasDefault {
		// Create a dummy block for the non-existent default case. It'll just get optimized out later.
		defaultBlock = mlir.BlockCreate2(nil, nil)
		buildBlock(ctx, defaultBlock, func() {
			// Branch to the successor block.
			brOp := mlir.GoCreateBranchOperation(b.ctx, successor, nil, b._noLoc)
			appendOperation(ctx, brOp)
		})
		appendBlock(ctx, defaultBlock)
	}

	// Create the select operation.
	op := mlir.GoCreateChanSelectOp(b.ctx, hasDefault, sendArr, chans, defaultBlock, successor, bodyBlocks, b.location(stmt.Pos()))
	appendOperation(ctx, op)

	// Continue emission in the successor block.
	appendBlock(ctx, successor)
	setCurrentBlock(ctx, successor)
}

func (b *Builder) emitReceiveExpression(ctx context.Context, expr *ast.UnaryExpr) []mlir.Value {
	loc := b.location(expr.Pos())

	// Get the channel type.
	chanType := b.typeOf(ctx, expr.X).(*types.Chan)

	// Get the element type of the channel.
	elementType := b.GetStoredType(ctx, chanType.Elem())

	// Evaluate the channel over which the value will be sent.
	channel := b.emitExpr(ctx, expr.X)[0]

	var resultT []mlir.Type
	resultType := b.typeOf(ctx, expr)
	if _, ok := resultType.(*types.Tuple); ok {
		resultT = b.types(elementType, b.i1)
	} else {
		resultT = b.types(elementType)
	}

	// Emit the channel receive operation.
	op := mlir.GoCreateChanRecvOp(b.ctx, resultT, channel, loc)
	appendOperation(ctx, op)
	return resultsOf(op)
}

func (b *Builder) emitSendStatement(ctx context.Context, stmt *ast.SendStmt) {
	loc := b.location(stmt.Pos())

	// Evaluate the channel over which the value will be sent.
	channel := b.emitExpr(ctx, stmt.Chan)[0]

	// Evaluate the value to send.
	value := b.emitExpr(ctx, stmt.Value)[0]

	// Emit the channel send operation.
	op := mlir.GoCreateChanSendOp(b.ctx, channel, value, loc)
	appendOperation(ctx, op)
}

func (b *Builder) emitChanRange(ctx context.Context, stmt *ast.RangeStmt) {
	location := b.location(stmt.Pos())
	endLocation := b.location(stmt.End())
	chanType := b.typeOf(ctx, stmt.X).(*types.Chan)
	elementType := b.GetStoredType(ctx, chanType.Elem())
	//elementPtrType := b.pointerOf(ctx, chanType.Elem())

	// Create the exit block where execution will continue following the range statement.
	exitBlock := mlir.BlockCreate2(nil, nil)

	// Create all blocks involved with the for loop.
	rangeBlock := mlir.BlockCreate2(nil, nil)
	appendBlock(ctx, rangeBlock)

	bodyBlock := mlir.BlockCreate2(b.types(elementType), b.locations(b._noLoc))
	appendBlock(ctx, bodyBlock)

	// Any break statement immediately branch to the exit block.
	ctx = newContextWithSuccessorBlock(ctx, exitBlock)

	// Any continue statement should branch to the condition block.
	ctx = newContextWithPredecessorBlock(ctx, rangeBlock)

	// Evaluate the chan value that will be iterated over.
	X := b.emitExpr(ctx, stmt.X)[0]

	// Branch to the condition block from the current block.
	brOp := mlir.GoCreateBranchOperation(b.ctx, rangeBlock, nil, location)
	appendOperation(ctx, brOp)

	// The range operation must be emitted into a block by itself.
	buildBlock(ctx, rangeBlock, func() {
		// Emit the channel range operation.
		op := mlir.GoCreateChanRangeOp(b.ctx, X, bodyBlock, exitBlock, location)
		appendOperation(ctx, op)
	})

	// Build the loop body block.
	buildBlock(ctx, bodyBlock, func() {
		// The received value is passed through the first block argument.
		value := mlir.BlockGetArgument(bodyBlock, 0)

		// TODO: A load here might make more sense if the block argument can always be a pointer. Implement the other
		//       ranges this way first.
		/*
			// Load the element from the pointer returned by the range operation.
			loadOp := mlir.GoCreateLoadOperation(b.ctx, value, elementType, location)
			appendOperation(ctx, loadOp)
			value = resultOf(loadOp)
		*/

		if stmt.Value != nil {
			var recvValue Value
			if stmt.Tok == token.DEFINE {
				// A copy should be emitted into this block. Heap escape analysis should handle converting the stack
				// allocation to a heap allocation in the event that the loop variable escapes the current scope.
				copyAddr := b.makeCopyOf(ctx, value, b.location(stmt.Body.Pos()))
				recvValue = b.NewTempValue(copyAddr)
				b.setAddr(ctx, stmt.Value.(*ast.Ident), recvValue)
			} else {
				// Store the received value into the receiver var.
				recvValue = b.valueOf(ctx, stmt.Value)
				recvValue.Store(ctx, value, b.location(stmt.Body.Pos()))
			}
		}

		// Emit the loop body
		for _, stmt := range stmt.Body.List {
			b.emitStmt(ctx, stmt)
		}

		if !blockHasTerminator(currentBlock(ctx)) {
			// Branch to the post iteration block.
			brOp := mlir.GoCreateBranchOperation(b.ctx, rangeBlock, nil, endLocation)
			appendOperation(ctx, brOp)
		}
	})

	// Continue emission in the successor block.
	appendBlock(ctx, exitBlock)
	setCurrentBlock(ctx, exitBlock)
}
