package ssa

import (
	"context"
	"go/token"

	"go/ast"
	"go/types"

	"omibyte.io/sigo/mlir"
)

func (b *Builder) emitSelectStatement(ctx context.Context, stmt *ast.SelectStmt) {
	defaultIdx := -1
	bodyBlocks := make([]mlir.Block, len(stmt.Body.List))
	chans := make([]*ast.Ident, len(stmt.Body.List))
	isSend := make([]bool, len(stmt.Body.List))

	// Create the successor block for this statement.
	successor := mlir.BlockCreate2(nil, nil)

	// Break statements should branch to this statement's immediate successor.
	ctx = newContextWithSuccessorBlock(ctx, successor)

	// Create the clause blocks.
	for i, clause := range stmt.Body.List {
		clause := clause.(*ast.CommClause)
		if clause.Comm == nil {
			defaultIdx = i
		} else {
			// Extract the specific channel involved in the case clause.
			switch stmt := clause.Comm.(type) {
			case *ast.AssignStmt:
				chans[i] = stmt.Rhs[0].(*ast.UnaryExpr).X.(*ast.Ident)
			case *ast.ExprStmt:
				chans[i] = stmt.X.(*ast.UnaryExpr).X.(*ast.Ident)
			case *ast.SendStmt:
				chans[i] = stmt.Chan.(*ast.Ident)
				isSend[i] = true
			}
		}

		// Create the body block
		bodyBlocks[i] = mlir.BlockCreate2(nil, nil)
		buildBlock(ctx, bodyBlocks[i], func() {
			if clause.Comm != nil {
				// Emit the clause statement.
				b.emitStmt(ctx, clause.Comm)
			}

			// Emit the body block statements.
			for _, stmt := range clause.Body {
				b.emitStmt(ctx, stmt)
			}

			// Branch to the successor block.
			brOp := mlir.GoCreateBranchOperation(b.ctx, successor, nil, b.location(clause.End()))
			appendOperation(ctx, brOp)
		})
	}

	// Create the input slices.
	constLenValue := b.emitConstInt(ctx, int64(len(stmt.Body.List)), b.si, b.location(stmt.Pos()))
	chanSliceArrOp := mlir.GoCreateAllocaOperation(b.ctx, b.ptr, b._chan, constLenValue, false, b.location(stmt.Pos()))
	appendOperation(ctx, chanSliceArrOp)
	chanSliceValue := b.emitConstSlice(ctx, resultOf(chanSliceArrOp), len(stmt.Body.List), b.location(stmt.Pos()))

	sendSliceArrOp := mlir.GoCreateAllocaOperation(b.ctx, b.ptr, b.i1, constLenValue, false, b.location(stmt.Pos()))
	appendOperation(ctx, sendSliceArrOp)
	sendSliceValue := b.emitConstSlice(ctx, resultOf(sendSliceArrOp), len(stmt.Body.List), b.location(stmt.Pos()))

	readySliceArrOp := mlir.GoCreateAllocaOperation(b.ctx, b.ptr, b.si, constLenValue, false, b.location(stmt.Pos()))
	appendOperation(ctx, readySliceArrOp)
	readySliceValue := b.emitConstSlice(ctx, resultOf(readySliceArrOp), len(stmt.Body.List), b.location(stmt.Pos()))

	// Create the runtime call to select a ready channel.
	hasDefaultValue := b.emitConstBool(ctx, defaultIdx != -1, b.location(stmt.Pos()))
	callOp := mlir.GoCreateRuntimeCallOperation(b.ctx, "runtime.channelSelect", []mlir.Type{b.si},
		[]mlir.Value{chanSliceValue, sendSliceValue, readySliceValue, hasDefaultValue}, b.location(stmt.Pos()))
	appendOperation(ctx, callOp)
	caseIdxValue := resultOf(callOp)

	// Create the clause evaluator blocks.
	lastCaseLoc := b.location(stmt.Pos())
	for i, clause := range stmt.Body.List {
		// Append the body block.
		appendBlock(ctx, bodyBlocks[i])

		// Skip creating a conditional branch for the default block.
		if i == defaultIdx {
			continue
		}

		lastCaseLoc = b.location(clause.Pos())

		// Create the successor block in which will compute the next case comparison.
		exprSuccessor := mlir.BlockCreate2(nil, nil)

		// Create a constant integer value representing the index for the current case.
		idxValue := b.emitConstInt(ctx, int64(i), b.si, b.location(clause.Pos()))

		// Compare the case index value against the integer value of the current case.
		cmpIOp := mlir.GoCreateCmpIOperation(b.ctx, b.i1, b.cmpIPredicate(token.EQL, false), caseIdxValue, idxValue,
			b.location(clause.Pos()))
		appendOperation(ctx, cmpIOp)

		// Conditionally branch to the case block if the values are equal. Otherwise, branch to the next expression
		// successor block.
		condBrOp := mlir.GoCreateCondBranchOperation(b.ctx, resultOf(cmpIOp), bodyBlocks[i], nil, exprSuccessor, nil,
			b.location(clause.Pos()))
		appendOperation(ctx, condBrOp)

		// Continue emission in the expression successor block.
		setCurrentBlock(ctx, exprSuccessor)

		// Append the expression successor block.
		appendBlock(ctx, exprSuccessor)
	}

	// NOTE: Should be at the empty final expression successor block.
	// Branch to the default case body or the successor block if there is no default.
	if defaultIdx != -1 {
		brOp := mlir.GoCreateBranchOperation(b.ctx, bodyBlocks[defaultIdx], nil, lastCaseLoc)
		appendOperation(ctx, brOp)
	} else {
		brOp := mlir.GoCreateBranchOperation(b.ctx, successor, nil, lastCaseLoc)
		appendOperation(ctx, brOp)
	}

	// Continue emission in the successor block.
	appendBlock(ctx, successor)
	setCurrentBlock(ctx, successor)
}

func (b *Builder) emitReceiveExpression(ctx context.Context, expr *ast.UnaryExpr) []mlir.Value {
	// Get the channel type.
	chanType := b.typeOf(ctx, expr.X).(*types.Chan)

	// Get the element type of the channel.
	elementType := b.GetStoredType(ctx, chanType.Elem())

	// Evaluate the channel over which the value will be sent.
	channel := b.emitExpr(ctx, expr.X)[0]

	// Allocate memory on the stack to store the received value to.
	addrOp := mlir.GoCreateAllocaOperation(b.ctx, b.ptr, elementType, nil, false, b.location(expr.Pos()))
	appendOperation(ctx, addrOp)
	addr := resultOf(addrOp)

	// Emit the runtime call to perform the channel receive.
	op := mlir.GoCreateRuntimeCallOperation(b.ctx, "runtime.channelReceive", []mlir.Type{b.i1}, []mlir.Value{channel, addr}, b.location(expr.Pos()))
	appendOperation(ctx, op)
	okValue := resultOf(op)

	// Load the received value.
	op = mlir.GoCreateLoadOperation(b.ctx, addr, elementType, b.location(expr.Pos()))
	appendOperation(ctx, op)
	value := resultOf(op)
	return []mlir.Value{value, okValue}
}

func (b *Builder) emitSendStatement(ctx context.Context, stmt *ast.SendStmt) {
	// Evaluate the channel over which the value will be sent.
	channel := b.emitExpr(ctx, stmt.Chan)[0]

	// Evaluate the value to send.
	value := b.emitExpr(ctx, stmt.Value)[0]

	// Take the address of the value since the runtime expects a pointer.
	valueAddr := b.makeCopyOf(ctx, value, b.location(stmt.Pos()))

	// Emit the runtime call to perform the channel send.
	op := mlir.GoCreateRuntimeCallOperation(b.ctx, "runtime.channelSend", nil, []mlir.Value{channel, valueAddr}, b.location(stmt.Pos()))
	appendOperation(ctx, op)
}
