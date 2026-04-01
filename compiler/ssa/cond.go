package ssa

import (
	"context"
	"go/ast"
	"go/token"
	"go/types"

	"pkg.si-go.dev/go-mlir/mlir"
	"pkg.si-go.dev/sigo/goir/binding/goir"
)

func (b *Builder) emitIfStatement(ctx context.Context, stmt *ast.IfStmt) {
	// Create the block that will be branched to if the condition is true.
	thenBlock := mlir.NewBlock(nil, nil)

	// Create the block that will be branched to if the condition is false.
	elseBlock := mlir.NewBlock(nil, nil)

	// Create the block that will be branched to following either condition.
	exitBlock := mlir.NewBlock(nil, nil)

	// Evaluate the init statement first.
	if stmt.Init != nil {
		b.emitStmt(ctx, stmt.Init)
	}

	// Evaluate the if-statement condition.
	condValue := b.emitExpr(ctx, stmt.Cond)[0]

	// Conditionally branch to either the then block of the else block.
	condBrOp := goir.NewCondBranchOperation(b.ctx, condValue, thenBlock, nil, elseBlock, nil,
		b.location(ctx, stmt.Cond.End()))
	appendOperation(ctx, condBrOp)

	// Build the then block.
	appendBlock(ctx, thenBlock)
	buildBlock(ctx, thenBlock, func() {
		b.emitBlock(ctx, stmt.Body)
		if !blockHasTerminator(currentBlock(ctx)) {
			// Branch to the exit block.
			brOp := goir.NewBranchOperation(b.ctx, exitBlock, nil, b.location(ctx, stmt.End()))
			appendOperation(ctx, brOp)
		}
	})

	// Build the else block.
	appendBlock(ctx, elseBlock)
	buildBlock(ctx, elseBlock, func() {
		// NOTE: An else condition is optional.
		if stmt.Else != nil {
			b.emitStmt(ctx, stmt.Else)
		}

		// NOTE: An if-statement may change the current block to a new one that is not the else-block.
		if !blockHasTerminator(currentBlock(ctx)) {
			// Branch to the exit block.
			brOp := goir.NewBranchOperation(b.ctx, exitBlock, nil, b.location(ctx, stmt.End()))
			appendOperation(ctx, brOp)
		}
	})

	// Append the exit block.
	appendBlock(ctx, exitBlock)

	// Continue emission in the exit block.
	setCurrentBlock(ctx, exitBlock)
}

func (b *Builder) emitExpressionSwitchStatement(ctx context.Context, stmt *ast.SwitchStmt) {
	// Evaluate the init statement first.
	if stmt.Init != nil {
		b.emitStmt(ctx, stmt.Init)
	}

	// Do no further emission if there are no clauses. The initializer is at least guaranteed to be emitted if one was
	// specified though.
	if len(stmt.Body.List) == 0 {
		// NOOP
		return
	}

	var tagValue mlir.ValueLike
	if stmt.Tag != nil {
		// Evaluate the tag expression
		tagValue = b.emitExpr(ctx, stmt.Tag)[0]
	}

	// Create the block where execution will be continued following the switch statement.
	exitBlock := mlir.NewBlock(nil, nil)

	// Create the done block where control flow will either branch to the default block body or the exit block.
	doneBlock := mlir.NewBlock(nil, nil)

	// Create all the case clause condition and body blocks.
	defaultBlock := -1
	bodyMap := map[int]int{}
	bodyBlocks := make([]mlir.Block, 0, len(stmt.Body.List))
	condBlocks := make([]mlir.Block, 0, len(stmt.Body.List))
	expressions := make([]ast.Expr, 0, len(stmt.Body.List))
	for i, clause := range stmt.Body.List {
		clause := clause.(*ast.CaseClause)
		if clause.List == nil {
			// NOTE: The default case does not have a condition block.
			defaultBlock = i
		} else {
			// A case may have more than a single clause expression, so create a new block for each.
			for _, expr := range clause.List {
				// Map this index to the expected clause body block. This eliminates the need for complicated
				// comparisons below.
				bodyMap[len(condBlocks)] = i

				// Append the expression to the expression slice for lookup later.
				expressions = append(expressions, expr)

				// Create the block where this case clause condition will be evaluated.
				condBlocks = append(condBlocks, mlir.NewBlock(nil, nil))
			}
		}

		// Create the block where the case clause body will be executed.
		bodyBlocks = append(bodyBlocks, mlir.NewBlock(nil, nil))
	}

	// Append the done block to the clause condition block slice so that it is jumped to last.
	condBlocks = append(condBlocks, doneBlock)

	// Branch to the first clause condition block to start.
	brOp := goir.NewBranchOperation(b.ctx, condBlocks[0], nil, b.location(ctx, stmt.Pos()))
	appendOperation(ctx, brOp)

	// Emit all the clause blocks stopping before the done block.
	for i, condBlock := range condBlocks[:len(condBlocks)-1] {
		// Emit the clause condition block.
		appendBlock(ctx, condBlock)
		buildBlock(ctx, condBlock, func() {
			expr := expressions[i]
			location := b.location(ctx, expr.Pos())
			bodyBlock := bodyBlocks[bodyMap[i]]

			// Evaluate the clause expression.
			value := b.emitExpr(ctx, expr)[0]

			if tagValue != nil {
				// Emit a comparison the result of the clause expression and the tag value.
				T := b.typeOf(ctx, expr)
				switch {
				case typeHasFlags(T, types.IsBoolean), typeHasFlags(T, types.IsInteger):
					value = b.emitIntegerCompare(ctx, token.EQL, value, tagValue, location)
				case typeHasFlags(T, types.IsFloat):
					value = b.emitFloatCompare(ctx, token.EQL, value, tagValue, location)
				case typeHasFlags(T, types.IsComplex):
					value = b.emitComplexCompare(ctx, token.EQL, value, tagValue, location)
				case typeHasFlags(T, types.IsString):
					value = b.emitStringCompare(ctx, token.EQL, value, tagValue, location)
				case isPointer(T):
					value = b.emitPointerCompare(ctx, token.EQL, value, tagValue, location)
				case typeIs[*types.Interface](T):
					value = b.emitInterfaceCompare(ctx, token.EQL, value, tagValue, location)
				case typeIs[*types.Struct](T):
					value = b.emitStructCompare(ctx, token.EQL, value, tagValue, baseType(T).(*types.Struct), location)
				default:
					panic("unhandled switch comparison operand type")
				}
			}

			// Conditionally branch to the respective clause body block (EXPR = true) or the next clause condition block
			//	(EXPR = false).
			condBrOp := goir.NewCondBranchOperation(b.ctx, value, bodyBlock, nil, condBlocks[i+1], nil,
				location)
			appendOperation(ctx, condBrOp)
		})
	}

	// Append the done block.
	appendBlock(ctx, doneBlock)

	// Emit the done block.
	buildBlock(ctx, doneBlock, func() {
		// Determine where the end of the clauses currently is.
		location := b.location(ctx, stmt.End())
		if lastIndex := len(stmt.Body.List) - 1; lastIndex > 0 {
			location = b.location(ctx, stmt.Body.List[lastIndex].End())
		}

		// NOTE: The position of the default block in the body block slice would have been determined earlier when all
		//       clause blocks were created.
		if defaultBlock >= 0 {
			// Branch to the default body block.
			brOp := goir.NewBranchOperation(b.ctx, bodyBlocks[defaultBlock], nil, location)
			appendOperation(ctx, brOp)
		} else {
			// Branch to the exit block.
			brOp := goir.NewBranchOperation(b.ctx, exitBlock, nil, location)
			appendOperation(ctx, brOp)
		}
	})

	// Emit all body blocks.
	for i, clause := range stmt.Body.List {
		scope := currentInfo(ctx).Scopes[clause]
		ctx := newContextWithScope(ctx, b.scopeAttr(scope, currentScope(ctx)))

		// Any break statement should immediately branch to the exit block.
		ctx = newContextWithSuccessorBlock(ctx, exitBlock, nil)

		if i+1 < len(stmt.Body.List) {
			// Fallthrough should go the next block.
			ctx = newContextWithFallthroughBlock(ctx, bodyBlocks[i+1], nil)
		}

		clause := clause.(*ast.CaseClause)
		bodyBlock := bodyBlocks[i]

		// Append the body block.
		appendBlock(ctx, bodyBlock)

		// Build the body block.
		buildBlock(ctx, bodyBlock, func() {
			// Emit clause body statements.
			b.emitStatements(ctx, clause.Body)

			// Some statement could have created a different terminator (IE panic, etc...)
			if !blockHasTerminator(currentBlock(ctx)) {
				// Branch to the exit block.
				brOp := goir.NewBranchOperation(b.ctx, exitBlock, nil, b.location(ctx, clause.End()))
				appendOperation(ctx, brOp)
			}
		})
	}

	// Append the exit block.
	appendBlock(ctx, exitBlock)

	// Continue emission in the exit block.
	setCurrentBlock(ctx, exitBlock)
}

func (b *Builder) emitForStatement(ctx context.Context, stmt *ast.ForStmt) {
	// Create the loop header block where the loop condition will be evaluated.
	headerBlock := mlir.NewBlock(nil, nil)

	// Create the body block where the loop body will be executed.
	bodyBlock := mlir.NewBlock(nil, nil)

	postIterationBlock := headerBlock
	if stmt.Post != nil {
		// Create the block where the post iteration statement will be executed.
		postIterationBlock = mlir.NewBlock(nil, nil)
	}

	// Create the exit block where execution should continue following the loop.
	exitBlock := mlir.NewBlock(nil, nil)

	// Evaluate the init statement first.
	if stmt.Init != nil {
		b.emitStmt(ctx, stmt.Init)
	}

	// Branch to the loop header block to start the loop.
	brOp := goir.NewBranchOperation(b.ctx, headerBlock, nil, b.location(ctx, stmt.Pos()))
	appendOperation(ctx, brOp)

	// Emit the header block.
	appendBlock(ctx, headerBlock)
	buildBlock(ctx, headerBlock, func() {
		// NOTE: The loop condition can be omitted which results in an infinite loop.
		if stmt.Cond != nil {
			// Evaluate the loop condition.
			condValue := b.emitExpr(ctx, stmt.Cond)[0]

			// Conditionally branch to either the body block or the exit block.
			condBrOp := goir.NewCondBranchOperation(b.config.Ctx, condValue, bodyBlock, nil, exitBlock, nil,
				b.location(ctx, stmt.Cond.Pos()))
			appendOperation(ctx, condBrOp)
		} else {
			// Unconditionally branch to the body block.
			brOp := goir.NewBranchOperation(b.ctx, bodyBlock, nil, b.location(ctx, stmt.Pos()))
			appendOperation(ctx, brOp)
		}
	})

	// Emit the loop body block.
	appendBlock(ctx, bodyBlock)
	buildBlock(ctx, bodyBlock, func() {
		// The continue statement should immediately branch to the post iteration block.
		// NOTE: The post iteration block may be a dedicated block or the header block if no post-iteration expression is
		//       present.
		ctx = newContextWithPredecessorBlock(ctx, postIterationBlock, nil)

		// The break statement should immediately branch to the exit block.
		ctx = newContextWithSuccessorBlock(ctx, exitBlock, nil)

		b.emitBlock(ctx, stmt.Body)
		if !blockHasTerminator(currentBlock(ctx)) {
			// Branch to the post-iteration block.
			// NOTE: This may be directly to the header block if there is no post-iteration expression.
			brOp := goir.NewBranchOperation(b.ctx, postIterationBlock, nil, b.location(ctx, stmt.Body.End()))
			appendOperation(ctx, brOp)
		}
	})

	// NOTE: The post-iteration statement is optional.
	if stmt.Post != nil {
		// Emit the post iteration block.
		appendBlock(ctx, postIterationBlock)
		buildBlock(ctx, postIterationBlock, func() {
			b.emitStmt(ctx, stmt.Post)

			// Branch to the header block.
			brOp := goir.NewBranchOperation(b.ctx, headerBlock, nil, b.location(ctx, stmt.Post.Pos()))
			appendOperation(ctx, brOp)
		})
	}

	// Append the exit block.
	appendBlock(ctx, exitBlock)

	// Continue emission in the exit block.
	setCurrentBlock(ctx, exitBlock)
}

func (b *Builder) emitRangeStatement(ctx context.Context, stmt *ast.RangeStmt) {
	T := b.typeOf(ctx, stmt.X)
	switch T := T.(type) {
	case *types.Array:
		b.emitArrayRange(ctx, stmt)
	case *types.Basic:
		if T.Kind() == types.String {
			b.emitStringRange(ctx, stmt)
		} else {
			b.emitIntRange(ctx, stmt)
		}
	case *types.Chan:
		b.emitChanRange(ctx, stmt)
	case *types.Map:
		b.emitMapRange(ctx, stmt)
	case *types.Slice:
		b.emitSliceRange(ctx, stmt)
	default:
		panic("unhandled")
	}
}

func (b *Builder) emitIntRange(ctx context.Context, stmt *ast.RangeStmt) {
	location := b.location(ctx, stmt.Pos())
	endLocation := b.location(ctx, stmt.End())
	tokLocation := b.location(ctx, stmt.TokPos)

	elementType := b.typeOf(ctx, stmt.X).(*types.Basic)
	elementT := b.GetStoredType(ctx, elementType)

	// Create the exit block where execution will continue following the range statement.
	exitBlock := mlir.NewBlock(nil, nil)

	// Create all blocks involved with the for loop.
	condBlock := mlir.NewBlock(b.types(b.si), b.locations(b._noLoc))
	appendBlock(ctx, condBlock)

	bodyBlock := mlir.NewBlock(b.types(elementT), b.locations(b._noLoc))
	appendBlock(ctx, bodyBlock)

	postIterBlock := mlir.NewBlock(b.types(elementT), b.locations(b._noLoc))
	appendBlock(ctx, postIterBlock)

	// Evaluate the upper limit of the range.
	X := b.emitExpr(ctx, stmt.X)[0]

	// NOTE: The value variable can be omitted from the range statement. There is no need to even consider the element
	//       value if no identifier to hold it is specified.
	valueVar := b.valueOf(ctx, stmt.Key)

	// Branch to the condition block from the current block.
	zeroValue := b.emitConstInt(ctx, 0, elementT, location)
	brOp := goir.NewBranchOperation(b.ctx, condBlock, b.values(zeroValue), location)
	appendOperation(ctx, brOp)

	// Build the condition block where the loop condition will continuously be evaluated in.
	buildBlock(ctx, condBlock, func() {
		value := condBlock.Argument(0)

		// Compare the iterator value against the array length value.
		cmpOp := goir.NewCmpIOperation(b.ctx, b.i1, b.cmpIPredicate(token.LSS, isUnsigned(elementT)), value, X, location)
		appendOperation(ctx, cmpOp)
		cond := resultOf(cmpOp)

		// Conditionally branch to the loop body block if the loop condition evaluates to true. Otherwise, branch to the
		// exit block.
		condBrOp := goir.NewCondBranchOperation(b.ctx, cond, bodyBlock, b.values(value), exitBlock, nil, location)
		appendOperation(ctx, condBrOp)
	})

	// Build the loop body block.
	buildBlock(ctx, bodyBlock, func() {
		value := bodyBlock.Argument(0)

		// Any break statement immediately branch to the exit block.
		ctx = newContextWithSuccessorBlock(ctx, exitBlock, nil)

		// Set the predecessor block to the post loop iteration block where any continue statement will branch to.
		ctx = newContextWithPredecessorBlock(ctx, postIterBlock, b.values(value))

		if stmt.Tok == token.DEFINE {
			ident := stmt.Key.(*ast.Ident)
			if identIsValid(ident) {
				// A copy should be emitted into this block. Heap escape analysis should handle converting the stack
				// allocation to a heap allocation in the event that the loop variable escapes the current scope.
				copyAddr := b.emitNamedAlloca(ctx, ident.Name, elementT, b.location(ctx, stmt.Key.Pos()))
				valueVar = b.NewTempValue(copyAddr)
				b.setAddr(ctx, ident, valueVar)
			}
		}

		// Store the loop variable values before executing the loop body.
		if valueVar != nil {
			// Store it at the element variable address.
			valueVar.Store(ctx, value, tokLocation)
		}

		// Emit the loop body.
		b.emitBlock(ctx, stmt.Body)

		if !blockHasTerminator(currentBlock(ctx)) {
			// Branch to the post iteration block.
			brOp := goir.NewBranchOperation(b.ctx, postIterBlock, b.values(value), endLocation)
			appendOperation(ctx, brOp)
		}
	})

	// Build the post iteration block.
	buildBlock(ctx, postIterBlock, func() {
		value := postIterBlock.Argument(0)

		// Increment the iterator value by one.
		oneValue := b.emitConstInt(ctx, 1, elementT, endLocation)
		addOp := goir.NewAddIOperation(b.ctx, elementT, value, oneValue, endLocation)
		appendOperation(ctx, addOp)

		// branch to the condition block.
		brOp := goir.NewBranchOperation(b.ctx, condBlock, resultsOf(addOp), endLocation)
		appendOperation(ctx, brOp)
	})

	// Continue emission in the successor block.
	appendBlock(ctx, exitBlock)
	setCurrentBlock(ctx, exitBlock)
}

func (b *Builder) emitTypeSwitchStatement(ctx context.Context, stmt *ast.TypeSwitchStmt) {
	// Create the successor block for this statement.
	successor := mlir.NewBlock(nil, nil)

	// Evaluate the init statement first.
	if stmt.Init != nil {
		b.emitStmt(ctx, stmt.Init)
	}

	// Get the interface value and/or the storage location for the resulting interface.
	var ifaceValue mlir.ValueLike
	var typeAssertExpr *ast.TypeAssertExpr
	switch assign := stmt.Assign.(type) {
	case *ast.ExprStmt:
		typeAssertExpr = ast.Unparen(assign.X).(*ast.TypeAssertExpr)
	case *ast.AssignStmt:
		typeAssertExpr = ast.Unparen(assign.Rhs[0]).(*ast.TypeAssertExpr)
	}

	ifaceValue = b.emitExpr(ctx, typeAssertExpr.X)[0]

	// Bitcast the interface value to the runtime interface type so it matches the body block argument type.
	ifaceForBlock := b.bitcastTo(ctx, ifaceValue, b._interface, b.location(ctx, typeAssertExpr.Pos()))

	// Create the clause blocks.
	defaultIdx := -1
	bodyBlocks := make([]mlir.Block, len(stmt.Body.List))
	for i, clause := range stmt.Body.List {
		clause := clause.(*ast.CaseClause)

		// NOTE: List is nil for the default clause.
		if clause.List == nil {
			defaultIdx = i
		}

		// Create the body block
		bodyBlocks[i] = mlir.NewBlock([]mlir.TypeLike{b._interface}, []mlir.LocationLike{b.location(ctx, clause.Pos())})
		buildBlock(ctx, bodyBlocks[i], func() {
			if obj := b.objectOf(ctx, clause); obj != nil {
				var local *LocalValue
				location := b.location(ctx, obj.Pos())
				value := bodyBlocks[i].Argument(0).AsValue()
				if len(clause.List) == 1 {
					// Extract the underlying pointer from the interface value.
					extractOp := goir.NewExtractOperation(b.ctx, 0, b.ptr, value, location)
					appendOperation(ctx, extractOp)

					// Load the concrete value.
					assertedType := b.GetStoredType(ctx, obj.Type())
					value = b.emitLoad(ctx, resultOf(extractOp), assertedType, location)

					// Allocate local storage for the asserted value.
					local = b.emitLocalVar(ctx, obj, assertedType, false)

					// Store the asserted value.
					local.Store(ctx, value, location)
				} else {
					// Bitcast to the "any" interface type.
					value = b.bitcastTo(ctx, value, b._any, location)

					// Allocate local storage for the interface value.
					local = b.emitLocalVar(ctx, obj, b._any, false)

					// Store the interface value.
					local.Store(ctx, value, location)
				}
			}

			scope := currentInfo(ctx).Scopes[clause]
			ctx := newContextWithScope(ctx, b.scopeAttr(scope, currentScope(ctx)))

			// Emit the body block statements.
			b.emitStatements(ctx, clause.Body)

			// Branch to the successor block only if the body doesn't already have a terminator (e.g. return).
			if !blockHasTerminator(currentBlock(ctx)) {
				brOp := goir.NewBranchOperation(b.ctx, successor, nil, b.location(ctx, clause.End()))
				appendOperation(ctx, brOp)
			}
		})
	}

	// Create the clause evaluator blocks.
	lastCaseLoc := b.location(ctx, stmt.Pos())
	for i, clause := range stmt.Body.List {
		clause := clause.(*ast.CaseClause)

		// Append the body block.
		appendBlock(ctx, bodyBlocks[i])

		// Skip creating a conditional branch for the default block.
		if i == defaultIdx {
			continue
		}

		for _, expr := range clause.List {
			lastCaseLoc = b.location(ctx, clause.Pos())

			// Create the successor block in which will compute the next case comparison.
			exprSuccessor := mlir.NewBlock(nil, nil)

			// Emit the operation to perform the type assertion.
			assertedT := b.GetType(ctx, b.typeOf(ctx, expr))
			op := goir.NewTypeAssertOperation(b.ctx, ifaceValue, b.types(assertedT, b.i1), b.location(ctx, expr.Pos()))
			appendOperation(ctx, op)
			results := resultsOf(op)

			// Conditionally branch to the body block if the type assertion was successful. Otherwise, branch to the
			// next expression evaluator block. Pass the bitcasted interface value since the body block expects
			// a runtime._interface-typed argument.
			condBrOp := goir.NewCondBranchOperation(b.ctx, results[1], bodyBlocks[i], []mlir.ValueLike{ifaceForBlock}, exprSuccessor, nil,
				b.location(ctx, clause.Pos()))
			appendOperation(ctx, condBrOp)

			// Continue emission in the expression successor block.
			setCurrentBlock(ctx, exprSuccessor)

			// Append the expression successor block.
			appendBlock(ctx, exprSuccessor)
		}
	}

	// NOTE: Should be at the empty final expression successor block.
	// Branch to the default case body or the successor block if there is no default.
	if defaultIdx != -1 {
		brOp := goir.NewBranchOperation(b.ctx, bodyBlocks[defaultIdx], []mlir.ValueLike{ifaceForBlock}, lastCaseLoc)
		appendOperation(ctx, brOp)
	} else {
		brOp := goir.NewBranchOperation(b.ctx, successor, nil, lastCaseLoc)
		appendOperation(ctx, brOp)
	}

	// Continue emission in the successor block.
	appendBlock(ctx, successor)
	setCurrentBlock(ctx, successor)
}
