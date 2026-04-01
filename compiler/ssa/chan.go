package ssa

import (
	"context"
	"go/token"

	"go/ast"
	"go/types"

	"pkg.si-go.dev/go-mlir/mlir"
	"pkg.si-go.dev/sigo/goir/binding/goir"
)

func (b *Builder) emitSelectStatement(ctx context.Context, stmt *ast.SelectStmt) {
	bodyBlocks := make([]mlir.Block, 0, len(stmt.Body.List))
	chans := make([]mlir.ValueLike, 0, len(stmt.Body.List))
	isSend := make([]bool, 0, len(stmt.Body.List))

	var defaultBlock mlir.Block
	hasDefault := false

	// Create the successor block for this statement.
	successor := mlir.NewBlock(nil, nil)

	// Create the clause blocks.
	for _, clause := range stmt.Body.List {
		clause := clause.(*ast.CommClause)
		var block mlir.Block
		if clause.Comm == nil {
			hasDefault = true
			defaultBlock = mlir.NewBlock(nil, nil)
			block = defaultBlock
		} else {
			// Extract the specific channel involved in the case clause.
			var value mlir.ValueLike
			var chanExpr ast.Expr
			send := false
			switch stmt := clause.Comm.(type) {
			case *ast.AssignStmt:
				chanExpr = stmt.Rhs[0].(*ast.UnaryExpr).X
				value = b.emitExpr(ctx, chanExpr)[0]
			case *ast.ExprStmt:
				chanExpr = stmt.X.(*ast.UnaryExpr).X
				value = b.emitExpr(ctx, chanExpr)[0]
			case *ast.SendStmt:
				chanExpr = stmt.Chan
				value = b.emitExpr(ctx, chanExpr)[0]
				send = true
			}

			// All channel directions are identical at runtime. If the channel
			// is directional, bitcast it to the bidirectional (SendRecv) type
			// so the variadic channel array has a uniform element type.
			if chanT, ok := b.typeOf(ctx, chanExpr).Underlying().(*types.Chan); ok && chanT.Dir() != types.SendRecv {
				sendRecvT := goir.NewChanType(
					b.GetStoredType(ctx, chanT.Elem()),
					goir.ChanDirectionSendRecv,
				)
				value = b.bitcastTo(ctx, value, sendRecvT, b.location(ctx, chanExpr.Pos()))
			}

			chans = append(chans, value)
			isSend = append(isSend, send)
			block = mlir.NewBlock(nil, nil)
			bodyBlocks = append(bodyBlocks, block)
		}

		// Create the body block
		buildBlock(ctx, block, func() {
			// Break statements should branch to this statement's immediate successor.
			ctx = newContextWithSuccessorBlock(ctx, successor, nil)

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
				brOp := goir.NewBranchOperation(b.ctx, successor, nil, b.location(ctx, clause.End()))
				appendOperation(ctx, brOp)
			}
		})
		appendBlock(ctx, block)
	}

	sendArr := mlir.NewDenseBoolArrayAttr(b.ctx, isSend)
	if !hasDefault {
		// Create a dummy block for the non-existent default case. It'll just get optimized out later.
		defaultBlock = mlir.NewBlock(nil, nil)
		buildBlock(ctx, defaultBlock, func() {
			// Branch to the successor block.
			brOp := goir.NewBranchOperation(b.ctx, successor, nil, b._noLoc)
			appendOperation(ctx, brOp)
		})
		appendBlock(ctx, defaultBlock)
	}

	// Create the select operation.
	op := goir.NewChanSelectOp(b.ctx, hasDefault, sendArr, chans, defaultBlock, successor, bodyBlocks, b.location(ctx, stmt.Pos()))
	appendOperation(ctx, op)

	// Continue emission in the successor block.
	appendBlock(ctx, successor)
	setCurrentBlock(ctx, successor)
}

func (b *Builder) emitReceiveExpression(ctx context.Context, expr *ast.UnaryExpr) []mlir.ValueLike {
	loc := b.location(ctx, expr.Pos())

	// Get the channel type.
	chanType := b.typeOf(ctx, expr.X).(*types.Chan)

	// Get the element type of the channel.
	elementType := b.GetStoredType(ctx, chanType.Elem())

	// Evaluate the channel over which the value will be sent.
	channel := b.emitExpr(ctx, expr.X)[0]

	var resultT []mlir.TypeLike
	resultType := b.typeOf(ctx, expr)
	if _, ok := resultType.(*types.Tuple); ok {
		resultT = b.types(elementType, b.i1)
	} else {
		resultT = b.types(elementType)
	}

	// Emit the channel receive operation.
	op := goir.NewChanRecvOp(b.ctx, resultT, channel, loc)
	appendOperation(ctx, op)
	return resultsOf(op)
}

func (b *Builder) emitSendStatement(ctx context.Context, stmt *ast.SendStmt) {
	loc := b.location(ctx, stmt.Pos())

	// Evaluate the channel over which the value will be sent.
	channel := b.emitExpr(ctx, stmt.Chan)[0]

	// Evaluate the value to send.
	value := b.emitExpr(ctx, stmt.Value)[0]

	// Emit the channel send operation.
	op := goir.NewChanSendOp(b.ctx, channel, value, loc)
	appendOperation(ctx, op)
}

func (b *Builder) emitChanRange(ctx context.Context, stmt *ast.RangeStmt) {
	location := b.location(ctx, stmt.Pos())
	endLocation := b.location(ctx, stmt.End())
	tokLocation := b.location(ctx, stmt.TokPos)

	chanType := b.typeOf(ctx, stmt.X).(*types.Chan)
	elementT := b.GetStoredType(ctx, chanType.Elem())

	// Create the exit block where execution will continue following the range statement.
	exitBlock := mlir.NewBlock(nil, nil)

	// Create all blocks involved with the for loop.
	rangeBlock := mlir.NewBlock(nil, nil)
	appendBlock(ctx, rangeBlock)

	bodyBlock := mlir.NewBlock(b.types(elementT), b.locations(b._noLoc))
	appendBlock(ctx, bodyBlock)

	// Evaluate the chan value that will be iterated over.
	X := b.emitExpr(ctx, stmt.X)[0]

	elementVar := b.valueOf(ctx, stmt.Value)

	// Branch to the condition block from the current block.
	brOp := goir.NewBranchOperation(b.ctx, rangeBlock, nil, location)
	appendOperation(ctx, brOp)

	// The range operation must be emitted into a block by itself.
	buildBlock(ctx, rangeBlock, func() {
		// Emit the channel range operation.
		op := goir.NewChanRangeOp(b.ctx, X, bodyBlock, exitBlock, location)
		appendOperation(ctx, op)
	})

	// Build the loop body block.
	buildBlock(ctx, bodyBlock, func() {
		value := bodyBlock.Argument(0)

		// Any break statement immediately branch to the exit block.
		ctx = newContextWithSuccessorBlock(ctx, exitBlock, nil)

		// Any continue statement should branch to the condition block.
		ctx = newContextWithPredecessorBlock(ctx, rangeBlock, nil)

		if stmt.Tok == token.DEFINE && stmt.Value != nil {
			ident := stmt.Value.(*ast.Ident)
			if identIsValid(ident) {
				// A copy should be emitted into this block. Heap escape analysis should handle converting the stack
				// allocation to a heap allocation in the event that the loop variable escapes the current scope.
				copyAddr := b.emitNamedAlloca(ctx, ident.Name, elementT, b.location(ctx, stmt.Value.Pos()))
				elementVar = b.NewTempValue(copyAddr)
				b.setAddr(ctx, ident, elementVar)
			}
		}

		if elementVar != nil {
			// Store it at the element variable address.
			elementVar.Store(ctx, value, tokLocation)
		}

		// Emit the loop body.
		b.emitBlock(ctx, stmt.Body)

		if !blockHasTerminator(currentBlock(ctx)) {
			// Branch to the post iteration block.
			brOp := goir.NewBranchOperation(b.ctx, rangeBlock, nil, endLocation)
			appendOperation(ctx, brOp)
		}
	})

	// Continue emission in the successor block.
	appendBlock(ctx, exitBlock)
	setCurrentBlock(ctx, exitBlock)
}
