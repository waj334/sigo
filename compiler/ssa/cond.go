package ssa

import (
	"context"
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/ast/astutil"

	"omibyte.io/sigo/mlir"
)

func (b *Builder) emitIfStatement(ctx context.Context, stmt *ast.IfStmt) {
	// Create the block that will be branched to if the condition is true.
	thenBlock := mlir.BlockCreate2(nil, nil)

	// Create the block that will be branched to if the condition is false.
	elseBlock := mlir.BlockCreate2(nil, nil)

	// Create the block that will be branched to following either condition.
	exitBlock := mlir.BlockCreate2(nil, nil)

	// Evaluate the init statement first.
	if stmt.Init != nil {
		b.emitStmt(ctx, stmt.Init)
	}

	// Evaluate the if-statement condition.
	condValue := b.emitExpr(ctx, stmt.Cond)[0]

	// Conditionally branch to either the then block of the else block.
	condBrOp := mlir.GoCreateCondBranchOperation(b.ctx, condValue, thenBlock, nil, elseBlock, nil,
		b.location(stmt.Pos()))
	appendOperation(ctx, condBrOp)

	// Build the then block.
	appendBlock(ctx, thenBlock)
	buildBlock(ctx, thenBlock, func() {
		b.emitBlock(ctx, stmt.Body)
		if !blockHasTerminator(currentBlock(ctx)) {
			// Branch to the exit block.
			brOp := mlir.GoCreateBranchOperation(b.ctx, exitBlock, nil, b.location(stmt.Body.End()))
			appendOperation(ctx, brOp)
		}
	})

	// Build the else block.
	appendBlock(ctx, elseBlock)
	buildBlock(ctx, elseBlock, func() {
		// NOTE: An else condition is optional.
		if stmt.Else != nil {
			// TODO: This switch can be eliminated once blocks are emitted by only the specific emitX functions.
			switch elseExpr := stmt.Else.(type) {
			case *ast.IfStmt:
				// Emit the else-if condition.
				b.emitIfStatement(ctx, elseExpr)
			case *ast.BlockStmt:
				// Emit the else block.
				b.emitBlock(ctx, elseExpr)
			default:
				panic("unhandled")
			}
		}

		// NOTE: An if-statement may change the current block to a new one that is not the else-block.
		if !blockHasTerminator(currentBlock(ctx)) {
			// Branch to the exit block.
			brOp := mlir.GoCreateBranchOperation(b.ctx, exitBlock, nil, b.location(stmt.Body.End()))
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

	var tagValue mlir.Value
	if stmt.Tag != nil {
		// Evaluate the tag expression
		tagValue = b.emitExpr(ctx, stmt.Tag)[0]
	}

	// Create the block where execution will be continued following the switch statement.
	exitBlock := mlir.BlockCreate2(nil, nil)

	// Create the done block where control flow will either branch to the default block body or the exit block.
	doneBlock := mlir.BlockCreate2(nil, nil)

	// Any break statement should immediately branch to the exit block.
	ctx = newContextWithSuccessorBlock(ctx, exitBlock)

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
				condBlocks = append(condBlocks, mlir.BlockCreate2(nil, nil))
			}
		}

		// Create the block where the case clause body will be executed.
		bodyBlocks = append(bodyBlocks, mlir.BlockCreate2(nil, nil))
	}

	// Append the done block to the clause condition block slice so that it is jumped to last.
	condBlocks = append(condBlocks, doneBlock)

	// Branch to the first clause condition block to start.
	brOp := mlir.GoCreateBranchOperation(b.ctx, condBlocks[0], nil, b.location(stmt.Pos()))
	appendOperation(ctx, brOp)

	// Emit all the clause blocks stopping before the done block.
	for i, condBlock := range condBlocks[:len(condBlocks)-1] {
		// Emit the clause condition block.
		appendBlock(ctx, condBlock)
		buildBlock(ctx, condBlock, func() {
			expr := expressions[i]
			location := b.location(expr.Pos())
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
					value = b.emitInterfaceCompare(ctx, token.EQL, value, tagValue, baseType(T), location)
				case typeIs[*types.Struct](T):
					value = b.emitStructCompare(ctx, token.EQL, value, tagValue, baseType(T).(*types.Struct), location)
				default:
					panic("unhandled switch comparison operand type")
				}
			}

			// Conditionally branch to the respective clause body block (EXPR = true) or the next clause condition block
			//	(EXPR = false).
			condBrOp := mlir.GoCreateCondBranchOperation(b.ctx, value, bodyBlock, nil, condBlocks[i+1], nil,
				location)
			appendOperation(ctx, condBrOp)
		})
	}

	// Append the done block.
	appendBlock(ctx, doneBlock)

	// Emit the done block.
	buildBlock(ctx, doneBlock, func() {
		// Determine where the end of the clauses currently is.
		location := b.location(stmt.End())
		if lastIndex := len(stmt.Body.List) - 1; lastIndex > 0 {
			location = b.location(stmt.Body.List[lastIndex].End())
		}

		// NOTE: The position of the default block in the body block slice would have been determined earlier when all
		//       clause blocks were created.
		if defaultBlock >= 0 {
			// Branch to the default body block.
			brOp := mlir.GoCreateBranchOperation(b.ctx, bodyBlocks[defaultBlock], nil, location)
			appendOperation(ctx, brOp)
		} else {
			// Branch to the exit block.
			brOp := mlir.GoCreateBranchOperation(b.ctx, exitBlock, nil, location)
			appendOperation(ctx, brOp)
		}
	})

	// Emit all body blocks.
	for i, clause := range stmt.Body.List {
		clause := clause.(*ast.CaseClause)
		bodyBlock := bodyBlocks[i]

		// Append the body block.
		appendBlock(ctx, bodyBlock)

		// Emit clause body statements.
		buildBlock(ctx, bodyBlock, func() {
			for _, stmt := range clause.Body {
				switch stmt := stmt.(type) {
				case *ast.BranchStmt:
					if stmt.Tok == token.FALLTHROUGH {
						// Immediately go to the next clause block.
						// NOTE: The checker does not allow the last clause to fallthrough, so that doesn't need to be
						//       accounted for here.
						// NOTE: Default clause is allowed to fallthrough to the next clause block, but the note above
						//       still applies.
						brOp := mlir.GoCreateBranchOperation(b.ctx, bodyBlocks[i+1], nil, b.location(stmt.Pos()))
						appendOperation(ctx, brOp)
						continue
					}
				}

				// All other statements get emitted normally.
				b.emitStmt(ctx, stmt)
			}

			// Some statement could have created a different terminator (IE panic, etc...)
			if !blockHasTerminator(currentBlock(ctx)) {
				// Branch to the exit block.
				brOp := mlir.GoCreateBranchOperation(b.ctx, exitBlock, nil, b.location(clause.End()))
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
	headerBlock := mlir.BlockCreate2(nil, nil)

	// Create the body block where the loop body will be executed.
	bodyBlock := mlir.BlockCreate2(nil, nil)

	postIterationBlock := headerBlock
	if stmt.Post != nil {
		// Create the block where the post iteration statement will be executed.
		postIterationBlock = mlir.BlockCreate2(nil, nil)
	}

	// Create the exit block where execution should continue following the loop.
	exitBlock := mlir.BlockCreate2(nil, nil)

	// The continue statement should immediately branch to the post iteration block.
	// NOTE: The post iteration block may be a dedicated block or the header block if no post-iteration expression is
	//       present.
	ctx = newContextWithPredecessorBlock(ctx, postIterationBlock)

	// The break statement should immediately branch to the exit block.
	ctx = newContextWithSuccessorBlock(ctx, exitBlock)

	// Evaluate the init statement first.
	if stmt.Init != nil {
		b.emitStmt(ctx, stmt.Init)
	}

	// Branch to the loop header block to start the loop.
	brOp := mlir.GoCreateBranchOperation(b.ctx, headerBlock, nil, b.location(stmt.Pos()))
	appendOperation(ctx, brOp)

	// Emit the header block.
	appendBlock(ctx, headerBlock)
	buildBlock(ctx, headerBlock, func() {
		// NOTE: The loop condition can be omitted which results in an infinite loop.
		if stmt.Cond != nil {
			// Evaluate the loop condition.
			condValue := b.emitExpr(ctx, stmt.Cond)[0]

			// Conditionally branch to either the body block or the exit block.
			condBrOp := mlir.GoCreateCondBranchOperation(b.config.Ctx, condValue, bodyBlock, nil, exitBlock, nil,
				b.location(stmt.Pos()))
			appendOperation(ctx, condBrOp)
		} else {
			// Unconditionally branch to the body block.
			brOp := mlir.GoCreateBranchOperation(b.ctx, bodyBlock, nil, b.location(stmt.Pos()))
			appendOperation(ctx, brOp)
		}
	})

	// Emit the loop body block.
	appendBlock(ctx, bodyBlock)
	buildBlock(ctx, bodyBlock, func() {
		b.emitBlock(ctx, stmt.Body)
		if !blockHasTerminator(currentBlock(ctx)) {
			// Branch to the post-iteration block.
			// NOTE: This may be directly to the header block if there is no post-iteration expression.
			brOp := mlir.GoCreateBranchOperation(b.ctx, postIterationBlock, nil, b.location(stmt.Body.End()))
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
			brOp := mlir.GoCreateBranchOperation(b.ctx, headerBlock, nil, b.location(stmt.Post.Pos()))
			appendOperation(ctx, brOp)
		})
	}

	// Append the exit block.
	appendBlock(ctx, exitBlock)

	// Continue emission in the exit block.
	setCurrentBlock(ctx, exitBlock)
}

func (b *Builder) emitRangeStatement(ctx context.Context, stmt *ast.RangeStmt) {
	location := b.location(stmt.Pos())
	T := b.typeOf(ctx, stmt.X)

	switch valueType := T.(type) {
	case *types.Array:
		// Get the allocation for the array being ranged over.
		value := b.valueOf(ctx, stmt.X)
		if value == nil {
			// A stack allocation needs to be created so that the array can be addressable.
			allocType := b.GetStoredType(ctx, T)
			allocaOp := mlir.GoCreateAllocaOperation(b.ctx, b.pointerOf(ctx, T), allocType, nil, false, location)
			appendOperation(ctx, allocaOp)

			value = LocalValue{
				ptr: resultOf(allocaOp),
				T:   allocType,
				b:   b,
			}

			// Evaluate the expression and store its result at the allocation address.
			v := b.emitExpr(ctx, stmt.X)[0]
			value.Store(ctx, v, location)
		}

		// The base address of the array is at the beginning of its memory allocation.
		ptr := value.Pointer(ctx, location)

		keyType := b.GetStoredType(ctx, b.typeOf(ctx, stmt.Key))
		elementType := b.GetStoredType(ctx, valueType.Elem())
		lenValue := b.emitConstInt(ctx, valueType.Len(), keyType, location)
		b.emitArrayRange(ctx, stmt.Key, stmt.Value, stmt.Tok, ptr, elementType, lenValue, stmt.Body, location)
	case *types.Basic:
		b.emitStringRange(ctx, stmt)
	case *types.Chan:
		b.emitChanRange(ctx, stmt)
	case *types.Map:
		b.emitMapRange(ctx, stmt)
	case *types.Slice:
		value := b.emitExpr(ctx, stmt.X)[0]
		keyType := b.typeOf(ctx, stmt.Key)
		elementType := b.GetStoredType(ctx, valueType.Elem())

		// Bitcast the slice value to its runtime representation.
		bitcastOp := mlir.GoCreateBitcastOperation(b.ctx, value, b._slice, location)
		appendOperation(ctx, bitcastOp)
		value = resultOf(bitcastOp)

		// Extract the array value from the slice.
		extractOp := mlir.GoCreateExtractOperation(b.ctx, 0, b.ptr, value, location)
		appendOperation(ctx, extractOp)
		arrValue := resultOf(extractOp)

		// Extract the array length value from the slice.
		extractOp = mlir.GoCreateExtractOperation(b.ctx, 1, b.si, value, location)
		appendOperation(ctx, extractOp)
		lenValue := resultOf(extractOp)

		// Convert the slice length type to the expected key type.
		lenValue = b.emitTypeConversion(ctx, lenValue, types.Typ[types.Int], keyType, location)

		// Range over the slice
		b.emitArrayRange(ctx, stmt.Key, stmt.Value, stmt.Tok, arrValue, elementType, lenValue, stmt.Body, location)
	default:
		panic("unhandled")
	}
}

func (b *Builder) emitArrayRange(ctx context.Context, key ast.Expr, value ast.Expr, tok token.Token, arrValue mlir.Value,
	elementType mlir.Type, lenValue mlir.Value, body *ast.BlockStmt, location mlir.Location) {
	// Create the exit block where execution will continue following the range statement.
	exitBlock := mlir.BlockCreate2(nil, nil)

	// Create all blocks involved with the for loop.
	condBlock := mlir.BlockCreate2(nil, nil)
	appendBlock(ctx, condBlock)

	bodyBlock := mlir.BlockCreate2(nil, nil)
	appendBlock(ctx, bodyBlock)

	postIterBlock := mlir.BlockCreate2(nil, nil)
	appendBlock(ctx, postIterBlock)

	// Any break statement immediately branch to the exit block.
	ctx = newContextWithSuccessorBlock(ctx, exitBlock)

	// Any continue statement should branch to the post iteration block.
	ctx = newContextWithPredecessorBlock(ctx, postIterBlock)

	// Create or evaluate the loop variables.
	keyType := b.GetStoredType(ctx, b.typeOf(ctx, key))

	// The key variable is either a new one or an existing one.
	keyVar := b.valueOf(ctx, key)
	if keyVar == nil {
		// Need to allocate memory for this variable.
		ptrT := mlir.GoCreatePointerType(keyType)
		allocaOp := mlir.GoCreateAllocaOperation(b.ctx, ptrT, keyType, nil, false, location)
		appendOperation(ctx, allocaOp)
		keyVar = b.NewTempValue(resultOf(allocaOp))
	}

	// Initialize the iterator to zero.
	zeroValue := b.emitConstInt(ctx, 0, keyType, location)
	keyVar.Store(ctx, zeroValue, location)

	// The value variable is either a new one or an existing one.
	// NOTE: The value variable can be omitted from the range statement.
	var valueVar Value
	if value != nil {
		valueVar = b.valueOf(ctx, value)
		if valueVar == nil {
			// Need to allocate memory for this variable.
			ptrT := mlir.GoCreatePointerType(elementType)
			allocaOp := mlir.GoCreateAllocaOperation(b.ctx, ptrT, elementType, nil, false, location)
			appendOperation(ctx, allocaOp)
			valueVar = b.NewTempValue(resultOf(allocaOp))
		}
	}

	// Build the post iteration block.
	buildBlock(ctx, postIterBlock, func() {
		// Load the current iterator value to increment.
		itValue := keyVar.Load(ctx, location)

		// Increment the iterator value by one.
		oneValue := b.emitConstInt(ctx, 1, keyType, location)
		addOp := mlir.GoCreateAddIOperation(b.ctx, keyType, itValue, oneValue, location)
		appendOperation(ctx, addOp)

		// Store the new iterator value at the stack address.
		keyVar.Store(ctx, resultOf(addOp), location)

		// branch to the condition block.
		brOp := mlir.GoCreateBranchOperation(b.ctx, condBlock, nil, location)
		appendOperation(ctx, brOp)
	})

	// Build the condition block where the loop condition will continuously be evaluated in.
	buildBlock(ctx, condBlock, func() {
		// Load the current iterator value to compare.
		itValue := keyVar.Load(ctx, location)

		// Compare the iterator value against the array length value.
		cmpOp := mlir.GoCreateCmpIOperation(b.ctx, b.i1, b.cmpIPredicate(token.LSS, isUnsigned(keyType)), itValue, lenValue, location)
		appendOperation(ctx, cmpOp)
		cond := resultOf(cmpOp)

		// Conditionally branch to the loop body block if the loop condition evaluates to true. Otherwise, branch to the
		// exit block.
		condBrOp := mlir.GoCreateCondBranchOperation(b.ctx, cond, bodyBlock, nil, exitBlock, nil, location)
		appendOperation(ctx, condBrOp)
	})

	// Build the loop body block.
	buildBlock(ctx, bodyBlock, func() {
		// Set the predecessor block to the post loop iteration block where any continue statement will branch to.
		ctx = newContextWithPredecessorBlock(ctx, postIterBlock)

		iterationKeyVar := keyVar
		iterationValueVar := valueVar

		// TODO: Need to check if the iterator was actually omitted "_".
		if tok == token.DEFINE && iterationKeyVar != nil {
			// Emit a new local variable that is unique to this iteration to store the key into.
			alloc := b.makeCopyOf(ctx, iterationKeyVar.Load(ctx, location), location)
			iterationKeyVar = b.NewTempValue(alloc)
			b.setAddr(ctx, key.(*ast.Ident), iterationKeyVar)
		}

		// NOTE: No store should be performed when the value var is omitted in the range statement.
		if iterationValueVar != nil {
			if tok == token.DEFINE {
				// Emit a new local variable that is unique to this iteration to store the value into.
				// TODO: Need to set the allocation name to that of the local variable definition.
				alloc := b.makeCopyOf(ctx, iterationValueVar.Load(ctx, location), location)
				iterationValueVar = b.NewTempValue(alloc)
				b.setAddr(ctx, value.(*ast.Ident), iterationValueVar)
			}

			// Load the current iterator value to store into the key address.
			itValue := iterationKeyVar.Load(ctx, location)

			// Get the address of the array element at the current iterator index.
			gepOp := mlir.GoCreateGepOperation2(b.ctx, arrValue, elementType, []any{itValue}, mlir.GoCreatePointerType(elementType), location)
			appendOperation(ctx, gepOp)

			// Load the value from the array and store it at the value address.
			loadOp := mlir.GoCreateLoadOperation(b.ctx, resultOf(gepOp), elementType, location)
			appendOperation(ctx, loadOp)
			iterationValueVar.Store(ctx, resultOf(loadOp), location)
		}

		// Emit the loop body.
		for _, stmt := range body.List {
			b.emitStmt(ctx, stmt)
		}

		if !blockHasTerminator(currentBlock(ctx)) {
			// Branch to the post iteration block.
			brOp := mlir.GoCreateBranchOperation(b.ctx, postIterBlock, nil, location)
			appendOperation(ctx, brOp)
		}
	})

	// Branch to the condition block from the current block.
	brOp := mlir.GoCreateBranchOperation(b.ctx, condBlock, nil, location)
	appendOperation(ctx, brOp)

	// Continue emission in the successor block.
	appendBlock(ctx, exitBlock)
	setCurrentBlock(ctx, exitBlock)
}

func (b *Builder) emitChanRange(ctx context.Context, stmt *ast.RangeStmt) {
	location := b.location(stmt.Pos())

	// Create the exit block where execution will continue following the range statement.
	exitBlock := mlir.BlockCreate2(nil, nil)

	// Create all blocks involved with the for loop.
	condBlock := mlir.BlockCreate2(nil, nil)
	appendBlock(ctx, condBlock)

	bodyBlock := mlir.BlockCreate2(nil, nil)
	appendBlock(ctx, bodyBlock)

	// Any break statement immediately branch to the exit block.
	ctx = newContextWithSuccessorBlock(ctx, exitBlock)

	// Any continue statement should branch to the condition block.
	ctx = newContextWithPredecessorBlock(ctx, condBlock)

	// Evaluate the chan value that will be iterated over.
	X := b.emitExpr(ctx, stmt.X)[0]

	// Reinterpret the chan value as its runtime type.
	bitcastOp := mlir.GoCreateBitcastOperation(b.ctx, X, b._chan, location)
	appendOperation(ctx, bitcastOp)
	X = resultOf(bitcastOp)

	// Get the memory address to receive a value from the channel in.
	var receiveValue Value
	var receiveAddr mlir.Value
	if receiveValue = b.valueOf(ctx, stmt.Value); receiveValue != nil && stmt.Value != nil {
		receiveAddr = receiveValue.Pointer(ctx, location)
	}

	if receiveAddr == nil {
		zeroOp := mlir.GoCreateZeroOperation(b.ctx, b.ptr, location)
		appendOperation(ctx, zeroOp)
		receiveAddr = resultOf(zeroOp)
	}

	// Build the condition block where the loop condition will continuously be evaluated in.
	buildBlock(ctx, condBlock, func() {
		// Create the runtime call to perform a receive on the channel.
		callOp := mlir.GoCreateRuntimeCallOperation(b.ctx, "runtime.channelReceive",
			[]mlir.Type{b.i1},
			[]mlir.Value{X, receiveAddr, b.emitConstBool(ctx, true, location)},
			location)
		appendOperation(ctx, callOp)
		okValue := resultOf(callOp)

		// Conditionally branch to the loop body block if the loop condition evaluates to true. Otherwise, branch to the
		// exit block.
		condBrOp := mlir.GoCreateCondBranchOperation(b.ctx, okValue, bodyBlock, nil, exitBlock, nil, location)
		appendOperation(ctx, condBrOp)
	})

	// Build the loop body block.
	buildBlock(ctx, bodyBlock, func() {
		// Set the predecessor block to the post loop iteration block where any continue statement will branch to.
		ctx = newContextWithPredecessorBlock(ctx, condBlock)

		if stmt.Tok == token.DEFINE && receiveValue != nil {
			// Emit a new local variable that is unique to this iteration to store the value into.
			alloc := b.makeCopyOf(ctx, receiveValue.Load(ctx, location), location)
			iterationReceiveValue := b.NewTempValue(alloc)
			b.setAddr(ctx, stmt.Value.(*ast.Ident), iterationReceiveValue)
		}

		// Emit the loop body
		for _, stmt := range stmt.Body.List {
			b.emitStmt(ctx, stmt)
		}

		if !blockHasTerminator(currentBlock(ctx)) {
			// Branch to the post iteration block.
			brOp := mlir.GoCreateBranchOperation(b.ctx, condBlock, nil, location)
			appendOperation(ctx, brOp)
		}
	})

	// Branch to the condition block from the current block.
	brOp := mlir.GoCreateBranchOperation(b.ctx, condBlock, nil, location)
	appendOperation(ctx, brOp)

	// Continue emission in the successor block.
	appendBlock(ctx, exitBlock)
	setCurrentBlock(ctx, exitBlock)
}

func (b *Builder) emitMapRange(ctx context.Context, stmt *ast.RangeStmt) {
	location := b.location(stmt.Pos())

	// Create the exit block where execution will continue following the range statement.
	exitBlock := mlir.BlockCreate2(nil, nil)

	// Create all blocks involved with the for loop.
	condBlock := mlir.BlockCreate2(nil, nil)
	appendBlock(ctx, condBlock)

	bodyBlock := mlir.BlockCreate2(nil, nil)
	appendBlock(ctx, bodyBlock)

	// Any break statement immediately branch to the exit block.
	ctx = newContextWithSuccessorBlock(ctx, exitBlock)

	// Any continue statement should branch to the condition block.
	ctx = newContextWithPredecessorBlock(ctx, condBlock)

	// Evaluate the map value that will be iterated over.
	X := b.emitExpr(ctx, stmt.X)[0]
	T := b.typeOf(ctx, stmt.X).(*types.Map)

	// Reinterpret the map value as its runtime type.
	bitcastOp := mlir.GoCreateBitcastOperation(b.ctx, X, b._map, location)
	appendOperation(ctx, bitcastOp)
	X = resultOf(bitcastOp)

	// Create the map iterator.
	_itType := b.config.Program.LookupType("runtime", "_mapIterator")
	iteratorType := b.GetStoredType(ctx, _itType)
	allocaOp := mlir.GoCreateAllocaOperation(b.ctx, mlir.GoCreatePointerType(iteratorType), iteratorType, nil, false, location)
	appendOperation(ctx, allocaOp)
	it := LocalValue{
		ptr: resultOf(allocaOp),
		T:   iteratorType,
		b:   b,
	}

	// Initialize the iterator.
	zeroOp := mlir.GoCreateZeroOperation(b.ctx, iteratorType, location)
	appendOperation(ctx, zeroOp)
	itValue := resultOf(zeroOp)

	insertOp := mlir.GoCreateInsertOperation(b.ctx, 0, X, itValue, iteratorType, location)
	appendOperation(ctx, insertOp)
	itValue = resultOf(insertOp)
	it.Store(ctx, resultOf(insertOp), location)

	var iterateResults []mlir.Value

	// Build the condition block where the loop condition will continuously be evaluated in.
	buildBlock(ctx, condBlock, func() {
		// Create the runtime call to perform the next iteration over the string.
		callOp := mlir.GoCreateRuntimeCallOperation(b.ctx, "runtime.mapRange",
			[]mlir.Type{b.i1, b.ptr, b.ptr},
			[]mlir.Value{it.Pointer(ctx, location)}, location)
		appendOperation(ctx, callOp)
		iterateResults = resultsOf(callOp)

		// Conditionally branch to the loop body block if the loop condition evaluates to true. Otherwise, branch to the
		// exit block.
		condBrOp := mlir.GoCreateCondBranchOperation(b.ctx, iterateResults[0], bodyBlock, nil, exitBlock, nil, location)
		appendOperation(ctx, condBrOp)
	})

	// Build the loop body block.
	buildBlock(ctx, bodyBlock, func() {
		// Set the predecessor block to the post loop iteration block where any continue statement will branch to.
		ctx = newContextWithPredecessorBlock(ctx, condBlock)

		// Load the key value.
		keyType := b.GetStoredType(ctx, T.Key())
		loadOp := mlir.GoCreateLoadOperation(b.ctx, iterateResults[1], keyType, location)
		appendOperation(ctx, loadOp)
		keyValue := resultOf(loadOp)

		// Assign the key/value values.
		if keyAddr := b.valueOf(ctx, stmt.Key); keyAddr != nil {
			keyAddr.Store(ctx, keyValue, location)

			if stmt.Tok == token.DEFINE {
				// Emit a new local variable that is unique to this iteration to store the key into.
				alloc := b.makeCopyOf(ctx, keyValue, location)
				iterationKeyVar := b.NewTempValue(alloc)
				b.setAddr(ctx, stmt.Key.(*ast.Ident), iterationKeyVar)
			}
		}

		if stmt.Value != nil {
			// Load the value value.
			valueType := b.GetStoredType(ctx, T.Elem())
			loadOp = mlir.GoCreateLoadOperation(b.ctx, iterateResults[2], valueType, location)
			appendOperation(ctx, loadOp)
			value := resultOf(loadOp)

			if valueAddr := b.valueOf(ctx, stmt.Value); valueAddr != nil {
				valueAddr.Store(ctx, value, location)

				if stmt.Tok == token.DEFINE {
					// Emit a new local variable that is unique to this iteration to store the key into.
					alloc := b.makeCopyOf(ctx, value, location)
					iterationValueVar := b.NewTempValue(alloc)
					b.setAddr(ctx, stmt.Value.(*ast.Ident), iterationValueVar)
				}
			}
		}

		// Emit the loop body
		for _, stmt := range stmt.Body.List {
			b.emitStmt(ctx, stmt)
		}

		if !blockHasTerminator(currentBlock(ctx)) {
			// Branch to the condition block.
			brOp := mlir.GoCreateBranchOperation(b.ctx, condBlock, nil, location)
			appendOperation(ctx, brOp)
		}
	})

	// Branch to the condition block from the current block.
	brOp := mlir.GoCreateBranchOperation(b.ctx, condBlock, nil, location)
	appendOperation(ctx, brOp)

	// Continue emission in the successor block.
	appendBlock(ctx, exitBlock)
	setCurrentBlock(ctx, exitBlock)
}

func (b *Builder) emitStringRange(ctx context.Context, stmt *ast.RangeStmt) {
	location := b.location(stmt.Pos())

	// Create the exit block where execution will continue following the range statement.
	exitBlock := mlir.BlockCreate2(nil, nil)

	// Create all blocks involved with the for loop.
	condBlock := mlir.BlockCreate2(nil, nil)
	appendBlock(ctx, condBlock)

	bodyBlock := mlir.BlockCreate2(nil, nil)
	appendBlock(ctx, bodyBlock)

	// Any break statement immediately branch to the exit block.
	ctx = newContextWithSuccessorBlock(ctx, exitBlock)

	// Any continue statement should branch to the condition block.
	ctx = newContextWithPredecessorBlock(ctx, condBlock)

	// The key variable is either a new one or an existing one.
	keyVar := b.valueOf(ctx, stmt.Key)
	if keyVar == nil {
		// Need to allocate memory for this variable.
		keyType := b.typeOf(ctx, stmt.Key)
		keyT := b.GetStoredType(ctx, keyType)
		ptrT := mlir.GoCreatePointerType(keyT)
		allocaOp := mlir.GoCreateAllocaOperation(b.ctx, ptrT, keyT, nil, false, location)
		appendOperation(ctx, allocaOp)
		keyVar = b.NewTempValue(resultOf(allocaOp))
	}

	// The value variable is either a new one or an existing one.
	// NOTE: The value variable can be omitted from the range statement.
	var valueVar Value
	if stmt.Value != nil {
		valueVar = b.valueOf(ctx, stmt.Value)
		if valueVar == nil {
			// Need to allocate memory for this variable.
			elementType := b.typeOf(ctx, stmt.Value)
			elementT := b.GetStoredType(ctx, elementType)
			ptrT := mlir.GoCreatePointerType(elementT)
			allocaOp := mlir.GoCreateAllocaOperation(b.ctx, ptrT, elementT, nil, false, location)
			appendOperation(ctx, allocaOp)
			valueVar = b.NewTempValue(resultOf(allocaOp))
		}
	}

	// Evaluate the string value that will be iterated over.
	X := b.emitExpr(ctx, stmt.X)[0]

	// Reinterpret the string value as its runtime type.
	bitcastOp := mlir.GoCreateBitcastOperation(b.ctx, X, b._string, location)
	appendOperation(ctx, bitcastOp)
	X = resultOf(bitcastOp)

	// Create the string iterator.
	_itType := b.config.Program.LookupType("runtime", "_stringIterator")
	iteratorType := b.GetStoredType(ctx, _itType)
	allocaOp := mlir.GoCreateAllocaOperation(b.ctx, mlir.GoCreatePointerType(iteratorType), iteratorType, nil, false, location)
	appendOperation(ctx, allocaOp)
	it := LocalValue{
		ptr: resultOf(allocaOp),
		T:   iteratorType,
		b:   b,
	}

	// Initialize the iterator.
	zeroOp := mlir.GoCreateZeroOperation(b.ctx, iteratorType, location)
	appendOperation(ctx, zeroOp)
	itValue := resultOf(zeroOp)

	insertOp := mlir.GoCreateInsertOperation(b.ctx, 0, X, itValue, iteratorType, location)
	appendOperation(ctx, insertOp)
	itValue = resultOf(insertOp)
	it.Store(ctx, resultOf(insertOp), location)

	var iterateResults []mlir.Value

	// Build the condition block where the loop condition will continuously be evaluated in.
	buildBlock(ctx, condBlock, func() {
		// Create the runtime call to perform the next iteration over the string.
		callOp := mlir.GoCreateRuntimeCallOperation(b.ctx, "runtime.stringRange",
			[]mlir.Type{b.i1, b.si, b.si32},
			[]mlir.Value{it.Pointer(ctx, location)}, location)
		appendOperation(ctx, callOp)
		iterateResults = resultsOf(callOp)

		// Conditionally branch to the loop body block if the loop condition evaluates to true. Otherwise, branch to the
		// exit block.
		condBrOp := mlir.GoCreateCondBranchOperation(b.ctx, iterateResults[0], bodyBlock, nil, exitBlock, nil, location)
		appendOperation(ctx, condBrOp)
	})

	// Build the loop body block.
	buildBlock(ctx, bodyBlock, func() {
		// Set the predecessor block to the post loop iteration block where any continue statement will branch to.
		ctx = newContextWithPredecessorBlock(ctx, condBlock)

		// Assign the key/value values.
		// NOTE: Key value may be nil if its identifier is "_"
		keyVar.Store(ctx, iterateResults[1], location)
		if stmt.Tok == token.DEFINE {
			// Emit a new local variable that is unique to this iteration to store the key into.
			alloc := b.makeCopyOf(ctx, iterateResults[1], location)
			iterationKeyVar := b.NewTempValue(alloc)
			b.setAddr(ctx, stmt.Key.(*ast.Ident), iterationKeyVar)
		}

		if stmt.Value != nil {
			valueVar.Store(ctx, iterateResults[2], location)
			if stmt.Tok == token.DEFINE {
				// Emit a new local variable that is unique to this iteration to store the key into.
				alloc := b.makeCopyOf(ctx, iterateResults[2], location)
				iterationValueVar := b.NewTempValue(alloc)
				b.setAddr(ctx, stmt.Value.(*ast.Ident), iterationValueVar)
			}
		}

		// Emit the loop body
		for _, stmt := range stmt.Body.List {
			b.emitStmt(ctx, stmt)
		}

		if !blockHasTerminator(currentBlock(ctx)) {
			// Branch to the post iteration block.
			brOp := mlir.GoCreateBranchOperation(b.ctx, condBlock, nil, location)
			appendOperation(ctx, brOp)
		}
	})

	// Branch to the condition block from the current block.
	brOp := mlir.GoCreateBranchOperation(b.ctx, condBlock, nil, location)
	appendOperation(ctx, brOp)

	// Continue emission in the successor block.
	appendBlock(ctx, exitBlock)
	setCurrentBlock(ctx, exitBlock)
}

func (b *Builder) emitTypeSwitchStatement(ctx context.Context, stmt *ast.TypeSwitchStmt) {
	// Create the successor block for this statement.
	successor := mlir.BlockCreate2(nil, nil)

	// Evaluate the init statement first.
	if stmt.Init != nil {
		b.emitStmt(ctx, stmt.Init)
	}

	// Get the interface value and/or the storage location for the resulting interface.
	var ifaceValue mlir.Value
	var typeAssertExpr *ast.TypeAssertExpr
	switch assign := stmt.Assign.(type) {
	case *ast.ExprStmt:
		typeAssertExpr = astutil.Unparen(assign.X).(*ast.TypeAssertExpr)
	case *ast.AssignStmt:
		typeAssertExpr = astutil.Unparen(assign.Rhs[0]).(*ast.TypeAssertExpr)
	}

	ifaceValue = b.emitExpr(ctx, typeAssertExpr.X)[0]

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
		bodyBlocks[i] = mlir.BlockCreate2([]mlir.Type{b._interface}, []mlir.Location{b.location(clause.Pos())})
		buildBlock(ctx, bodyBlocks[i], func() {
			if obj := b.objectOf(ctx, clause); obj != nil {
				var local *LocalValue
				location := b.location(obj.Pos())
				value := mlir.BlockGetArgument(bodyBlocks[i], 0)
				if len(clause.List) == 1 {
					// Extract the underlying pointer from the interface value.
					extractOp := mlir.GoCreateExtractOperation(b.ctx, 0, b.ptr, value, location)
					appendOperation(ctx, extractOp)

					// Load the concrete value.
					assertedType := b.GetStoredType(ctx, obj.Type())
					loadOp := mlir.GoCreateLoadOperation(b.ctx, resultOf(extractOp), assertedType, location)
					value = resultOf(loadOp)
					appendOperation(ctx, loadOp)

					// Allocate local storage for the asserted value.
					local = b.emitLocalVar(ctx, obj, assertedType)

					// Store the asserted value.
					local.Store(ctx, value, location)
				} else {
					// Bitcast to the "any" interface type.
					value = b.bitcastTo(ctx, value, b._any, location)

					// Allocate local storage for the interface value.
					local = b.emitLocalVar(ctx, obj, b._any)

					// Store the interface value.
					local.Store(ctx, value, location)
				}
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

	// Create the clause evaluator blocks.
	lastCaseLoc := b.location(stmt.Pos())
	for i, clause := range stmt.Body.List {
		clause := clause.(*ast.CaseClause)

		// Append the body block.
		appendBlock(ctx, bodyBlocks[i])

		// Skip creating a conditional branch for the default block.
		if i == defaultIdx {
			continue
		}

		for _, expr := range clause.List {
			lastCaseLoc = b.location(clause.Pos())

			// Create the successor block in which will compute the next case comparison.
			exprSuccessor := mlir.BlockCreate2(nil, nil)

			// Get the type information about the asserted type.
			assertedType := b.GetType(ctx, b.typeOf(ctx, expr))
			infoOp := mlir.GoCreateTypeInfoOperation(b.ctx, b.typeInfoPtr, assertedType, b.location(expr.Pos()))
			info := resultOf(infoOp)
			appendOperation(ctx, infoOp)

			// Create the runtime call to perform the type assertion.
			// TODO: Replace this runtime call with an explicit operation for type assertion.
			trueValue := b.emitConstBool(ctx, true, b.location(stmt.Pos()))
			callOp := mlir.GoCreateRuntimeCallOperation(b.ctx, "runtime.interfaceAssert", []mlir.Type{b._interface, b.i1},
				[]mlir.Value{ifaceValue, info, trueValue}, b.location(expr.Pos()))
			appendOperation(ctx, callOp)
			results := resultsOf(callOp)

			// Compare the ok value against true.
			cmpIOp := mlir.GoCreateCmpIOperation(b.ctx, b.i1, b.cmpIPredicate(token.EQL, false), results[1], trueValue,
				b.location(clause.Pos()))
			appendOperation(ctx, cmpIOp)

			// Conditionally branch to the body block if the type assertion was successful. Otherwise, branch to the
			// next expression evaluator block.
			condBrOp := mlir.GoCreateCondBranchOperation(b.ctx, resultOf(cmpIOp), bodyBlocks[i], []mlir.Value{results[0]}, exprSuccessor, nil,
				b.location(clause.Pos()))
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
