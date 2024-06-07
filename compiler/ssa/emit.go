package ssa

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"omibyte.io/sigo/mlir"
	"os"
	"path/filepath"
	"runtime/debug"
)

func (b *Builder) emitAssign(ctx context.Context, stmt *ast.AssignStmt) {
	location := b.location(stmt.Pos())
	switch stmt.Tok {
	case token.ADD_ASSIGN, token.SUB_ASSIGN, token.MUL_ASSIGN, token.QUO_ASSIGN, token.REM_ASSIGN, token.AND_ASSIGN,
		token.OR_ASSIGN, token.XOR_ASSIGN, token.SHL_ASSIGN, token.SHR_ASSIGN, token.AND_NOT_ASSIGN:
		b.emitCompoundAssign(ctx, stmt)
		return
	case token.ASSIGN, token.DEFINE:
		// Get the types of the LHS expressions to use for type inference for untyped types.
		lhsTypes := make([]types.Type, len(stmt.Lhs))
		for i, lhs := range stmt.Lhs {
			T := b.typeOf(ctx, lhs)
			lhsTypes[i] = T
		}

		rhsTypes := make([]types.Type, 0, len(stmt.Rhs))
		for _, rhs := range stmt.Rhs {
			T := b.typeOf(ctx, rhs)
			if tupleT, ok := T.(*types.Tuple); ok {
				for i := 0; i < tupleT.Len(); i++ {
					rhsTypes = append(rhsTypes, tupleT.At(i).Type())
				}
			} else {
				rhsTypes = append(rhsTypes, T)
			}
		}

		ctx = newContextWithLhsList(ctx, lhsTypes)
		lvals := make([]Value, len(stmt.Lhs))
		rvals := make([]mlir.Value, 0, len(stmt.Rhs))
		for i, lhs := range stmt.Lhs {
			var lval Value
			switch expr := stmt.Lhs[i].(type) {
			case *ast.Ident:
				if expr.Name == "_" {
					// Skip evaluating this expression.
					continue
				}
			}

			if stmt.Tok == token.DEFINE {
				// Local variables need to be emitted into the current block.
				lval = b.emitLocalVar(ctx, b.objectOf(ctx, lhs), b.GetStoredType(ctx, lhsTypes[i]))
			} else {
				// Memory to hold the value should have already been created. Acquire the address of the memory
				// location.
				lval = b.valueOf(ctx, lhs)
			}
			lvals[i] = lval
		}

		for i, rhss := range stmt.Rhs {
			ctx = newContextWithRhsIndex(ctx, i)
			rval := b.emitExpr(ctx, rhss)
			rvals = append(rvals, rval...)
		}

		if len(lvals) > len(rvals) {
			panic("len(lvals) > len(rvals)")
		}

		// Store the RHS values into the LHS addresses.
		// NOTE: There can be fewer LHS addresses than RHS values. Any extra RHS values will be ignored.
		for i := range lvals {
			var lhs Value
			if lvals[i] != nil {
				lhs = lvals[i]
			} else {
				switch expr := stmt.Lhs[i].(type) {
				case *ast.Ident:
					if expr.Name == "_" {
						// Skip attempting to store this value.
						continue
					}
				case *ast.IndexExpr:
					lhs = b.NewTempValue(b.emitIndexAddr(ctx, expr))
				case *ast.SelectorExpr:
					// Compute the address to store to.
					lhs = b.NewTempValue(b.emitSelectAddr(ctx, expr))
				case *ast.StarExpr:
					// The RHS value should be stored at the address of the variable.
					lhs = b.NewTempValue(b.emitExpr(ctx, expr.X)[0])
				default:
					// Evaluate the LHS.
					lhs = b.NewTempValue(b.emitExpr(ctx, stmt.Lhs[i])[0])
				}
			}

			if i < len(rvals) {
				rhs := rvals[i]
				lhsType := lhsTypes[i]
				switch baseType(lhsType).(type) {
				case *types.Interface:
					rhsType := rhsTypes[i]
					if !isNil(rhsType) && !types.Identical(lhsType, rhsType) {
						if types.IsInterface(baseType(rhsType)) {
							// Convert from interface A to interface B.
							rhs = b.emitChangeType(ctx, lhsType, rhs, location)
						} else {
							// Generate methods for named types.
							if T, ok := rhsType.(*types.Named); ok {
								b.queueNamedTypeJobs(ctx, T)
							}

							// Create an interface value from the value expression.
							rhs = b.emitInterfaceValue(ctx, lhsType, rhsType, rhs, location)
						}
					}
				}

				lhs.Store(ctx, rhs, location)
			}
		}
	default:
		panic("unhandled")
	}
}

func (b *Builder) emitCompoundAssign(ctx context.Context, stmt *ast.AssignStmt) {
	location := b.location(stmt.Pos())

	var op token.Token
	switch stmt.Tok {
	case token.ADD_ASSIGN:
		op = token.ADD
	case token.SUB_ASSIGN:
		op = token.SUB
	case token.MUL_ASSIGN:
		op = token.MUL
	case token.QUO_ASSIGN:
		op = token.QUO
	case token.REM_ASSIGN:
		op = token.REM
	case token.AND_ASSIGN:
		op = token.AND
	case token.OR_ASSIGN:
		op = token.OR
	case token.XOR_ASSIGN:
		op = token.XOR
	case token.SHL_ASSIGN:
		op = token.SHL
	case token.SHR_ASSIGN:
		op = token.SHR
	case token.AND_NOT_ASSIGN:
		op = token.AND_NOT
	default:
		panic("unhandled")
	}

	// Get the address of the LHS variable.
	lvar := b.valueOf(ctx, stmt.Lhs[0])
	if lvar == nil {
		switch node := stmt.Lhs[0].(type) {
		case *ast.SelectorExpr:
			lvar = b.valueOf(ctx, node.Sel)
		default:
			panic("unhandled")
		}
	}

	// Load the current value of the LHS variable.
	X := lvar.Load(ctx, location)

	// Set up type inference.
	lhsT := b.typeOf(ctx, stmt.Lhs[0])
	ctx = newContextWithLhsList(ctx, []types.Type{lhsT})
	ctx = newContextWithRhsIndex(ctx, 0)

	// Evaluate the RHS value.
	rhsT := b.typeOf(ctx, stmt.Rhs[0])
	Y := b.emitExpr(ctx, stmt.Rhs[0])[0]

	// Make sure operands are the same type.
	Y = b.emitTypeConversion(ctx, Y, rhsT, lhsT, location)

	// Perform the respective arithmetic operation.
	T := b.GetStoredType(ctx, lhsT)
	X = b.emitArith(ctx, op, X, Y, lhsT, T, location)

	// Store the result at the LHS address.
	lvar.Store(ctx, X, location)
}

func (b *Builder) emitBinaryExpression(ctx context.Context, expr *ast.BinaryExpr) mlir.Value {
	// Get the types of the LHS expressions to use for type inference for untyped types.
	// NOTE: Use the type of the left-most operand if consecutive right-hand operands are also untyped.
	var exprT types.Type
	lhsT := b.typeOf(ctx, expr.X)
	rhsT := b.typeOf(ctx, expr.Y)
	if isUntyped(lhsT) || isUntyped(rhsT) {
		if !isUntyped(lhsT) {
			exprT = lhsT
		} else if !isUntyped(rhsT) {
			exprT = rhsT
		}
	}

	if exprT == nil {
		exprT = b.typeOf(ctx, expr)
		if isUntyped(exprT) {
			exprT = resolveType(ctx, exprT)
		}
	}

	ctx = newContextWithLhsList(ctx, []types.Type{exprT})
	ctx = newContextWithRhsIndex(ctx, 0)

	// Create the respective binary expression operation.
	location := b.location(expr.Pos())
	switch expr.Op {
	case token.SHL, token.SHR:
		X := b.emitExpr(ctx, expr.X)[0]
		Y := b.emitExpr(ctx, expr.Y)[0]

		// Make sure operands are the same type.
		X = b.emitTypeConversion(ctx, X, lhsT, exprT, location)
		Y = b.emitTypeConversion(ctx, Y, rhsT, exprT, location)

		resultT := exprT

		// The terms in the expression must be integers!
		// NOTE: This is a special case for this expression: float64(r.Int63()) / (1 << 63).
		if !typeHasFlags(exprT, types.IsInteger) {
			X = b.emitTypeConversion(ctx, X, exprT, types.Typ[types.Int], location)
			Y = b.emitTypeConversion(ctx, Y, exprT, types.Typ[types.Int], location)
			exprT = types.Typ[types.Int]
		}

		// Emit the arithmetic operation.
		T := b.GetStoredType(ctx, exprT)
		result := b.emitArith(ctx, expr.Op, X, Y, exprT, T, location)

		if !types.Identical(resultT, exprT) {
			// Cast the result the original type (some float type).
			result = b.emitTypeConversion(ctx, result, exprT, resultT, location)
		}
		return result
	case token.ADD, token.SUB, token.MUL, token.QUO, token.REM, token.AND, token.OR, token.XOR, token.AND_NOT:
		T := b.GetStoredType(ctx, exprT)

		// Get the operand values to be used in the binary expression.
		X := b.emitExpr(ctx, expr.X)[0]
		Y := b.emitExpr(ctx, expr.Y)[0]

		// Make sure operands are the same type.
		X = b.emitTypeConversion(ctx, X, lhsT, exprT, location)
		Y = b.emitTypeConversion(ctx, Y, rhsT, exprT, location)

		// Emit the arithmetic operation.
		return b.emitArith(ctx, expr.Op, X, Y, exprT, T, location)
	case token.EQL, token.NEQ, token.GTR, token.LSS, token.LEQ, token.GEQ:
		return b.emitComparison(ctx, expr)
	case token.LAND, token.LOR:
		return b.emitLogicalComparison(ctx, expr)
	default:
		panic("unhandled binary expression " + expr.Op.String())
	}
}

func (b *Builder) emitBlock(ctx context.Context, stmt *ast.BlockStmt) {
	if stmt == nil {
		// There are no statements nested in this block.
		return
	}

	// Emit operations for every statement in the input block.
	for _, stmt := range stmt.List {
		// Do not continue emission if the current block is already terminated.
		if blockHasTerminator(currentBlock(ctx)) {
			return
		}

		b.emitStmt(ctx, stmt)
	}
}

func (b *Builder) emitBranchStatement(ctx context.Context, stmt *ast.BranchStmt) {
	predecessor := currentPredecessorBlock(ctx)
	successor := currentSuccessorBlock(ctx)
	switch stmt.Tok {
	case token.BREAK:
		block := successor
		if stmt.Label != nil {
			// Immediately branch to the specified predecessor block.
			block = currentLabeledBlocks(ctx)[stmt.Label.Name]
		} // Otherwise, branch to the successor block.
		brOp := mlir.GoCreateBranchOperation(b.ctx, block, nil, b.location(stmt.Pos()))
		appendOperation(ctx, brOp)
		return
	case token.GOTO:
		block := currentLabeledBlocks(ctx)[stmt.Label.Name]
		brOp := mlir.GoCreateBranchOperation(b.ctx, block, nil, b.location(stmt.Pos()))
		appendOperation(ctx, brOp)
		return
	case token.FALLTHROUGH:
		// The checker does not allow the fallthrough statement to appear in any block nested in a switch clause
		// body block.
		panic("unreachable")
	case token.CONTINUE:
		// Immediately branch to the predecessor block.
		brOp := mlir.GoCreateBranchOperation(b.ctx, predecessor, nil, b.location(stmt.Pos()))
		appendOperation(ctx, brOp)
	default:
		panic("unhandled switch branch statement")
	}
}

func (b *Builder) emitDecl(ctx context.Context, decl ast.Decl) {
	defer func() {
		if v := recover(); v != nil {
			pos := b.config.Fset.Position(decl.Pos())
			fname, _ := filepath.EvalSymlinks(pos.Filename)
			fname = fmt.Sprintf("%s:%d:%d", fname, pos.Line, pos.Column)
			fmt.Fprintf(os.Stderr, "failure while emitting %T: %+v\n%s\n\n%s\n",
				decl, v, fname, string(debug.Stack()))
			os.Exit(-1)
		}
	}()

	switch decl := decl.(type) {
	case *ast.FuncDecl:
		// Function declarations should NOT be generated here.
		panic("unreachable")
	case *ast.GenDecl:
		b.emitGenericDecl(ctx, decl)
	default:
		panic("unhandled declaration statement")
	}
}

func (b *Builder) emitExpr(ctx context.Context, expr ast.Expr) []mlir.Value {
	defer func() {
		if v := recover(); v != nil {
			pos := b.config.Fset.Position(expr.Pos())
			fname, _ := filepath.EvalSymlinks(pos.Filename)
			fname = fmt.Sprintf("%s:%d:%d", fname, pos.Line, pos.Column)
			line := b.locationString(expr.Pos())
			fmt.Fprintf(os.Stderr, "failure while emitting %T: %+v\n%s\n\n%s\n\n%s\n",
				expr, v, fname, line, string(debug.Stack()))
			os.Exit(-1)
		}
	}()

	switch expr := expr.(type) {
	case *ast.BasicLit:
		return []mlir.Value{b.emitBasicLiteral(ctx, expr)}
	case *ast.BinaryExpr:
		return []mlir.Value{b.emitBinaryExpression(ctx, expr)}
	case *ast.CallExpr:
		return b.emitCallExpr(ctx, expr)
	case *ast.CompositeLit:
		return []mlir.Value{b.emitCompositeLiteral(ctx, expr)}
	case *ast.Ellipsis:
		panic("unreachable")
	case *ast.FuncLit:
		return []mlir.Value{b.emitFuncLiteral(ctx, expr)}
	case *ast.Ident:
		return b.emitIdent(ctx, expr)
	case *ast.IndexExpr:
		return b.emitIndexExpr(ctx, expr)
	case *ast.IndexListExpr:
		panic("unreachable")
	case *ast.KeyValueExpr:
		panic("unreachable")
	case *ast.ParenExpr:
		exprT := b.typeOf(ctx, expr)
		if !typeHasFlags(exprT, types.IsUntyped) {
			// Update the inferred types.
			ctx = newContextWithLhsList(ctx, []types.Type{b.typeOf(ctx, expr)})
			ctx = newContextWithRhsIndex(ctx, 0)
		}
		return b.emitExpr(ctx, expr.X)
	case *ast.SelectorExpr:
		return b.emitSelectorExpr(ctx, expr)
	case *ast.SliceExpr:
		return b.emitSliceExpr(ctx, expr)
	case *ast.StarExpr:
		return b.emitStarExpr(ctx, expr)
	case *ast.TypeAssertExpr:
		return b.emitTypeAssertExpr(ctx, expr)
	case *ast.UnaryExpr:
		return b.emitUnaryExpr(ctx, expr)
	case *ast.ArrayType, *ast.ChanType, *ast.StructType, *ast.FuncType, *ast.InterfaceType, *ast.MapType:
		T := b.GetStoredType(ctx, b.typeOf(ctx, expr))
		op := mlir.GoCreateTypeInfoOperation(b.ctx, b.typeInfoPtr, T, b.location(expr.Pos()))
		appendOperation(ctx, op)
		return []mlir.Value{resultOf(op)}
	default:
		panic("unhandled expression statement")
	}
	return nil
}

func (b *Builder) emitGenericDecl(ctx context.Context, decl *ast.GenDecl) {
	switch decl.Tok {
	case token.CONST:
		b.emitConstantDecl(ctx, decl)
	case token.IMPORT:
		// Do nothing
	case token.TYPE:
		// Do nothing
	case token.VAR:
		for _, spec := range decl.Specs {
			spec := spec.(*ast.ValueSpec)
			vars := make([]Value, len(spec.Names))

			// Local variables need to be emitted into the current block.
			for i, ident := range spec.Names {
				vars[i] = b.emitLocalVar(ctx, b.objectOf(ctx, ident), b.GetStoredType(ctx, b.typeOf(ctx, ident)))
			}

			// Assign initial value (if any).
			for i, expr := range spec.Values {
				// Evaluate the initial value.
				result := b.emitExpr(ctx, expr)[0]

				// Store the initial value at the address of the variable.
				vars[i].Store(ctx, result, b.location(decl.Pos()))
			}
		}
	default:
		panic("invalid generic declaration")
	}
}

func (b *Builder) emitIdent(ctx context.Context, expr *ast.Ident) []mlir.Value {
	location := b.location(expr.Pos())
	obj := b.objectOf(ctx, expr)
	switch obj := obj.(type) {
	case *types.Const:
		exprType := b.typeOf(ctx, expr)
		T := resolveType(ctx, exprType)
		val := b.emitConstantValue(ctx, obj.Val(), T, location)
		return []mlir.Value{val}
	case *types.Func:
		symbol := b.resolveSymbol(qualifiedFuncName(obj))
		b.queueJob(ctx, symbol)
		signatureType := b.createSignatureType(ctx, obj.Type().Underlying().(*types.Signature), false)
		op := mlir.GoCreateAddressOfOperation(b.ctx, symbol, signatureType, location)
		appendOperation(ctx, op)
		return resultsOf(op)
	case *types.Nil:
		var T mlir.Type
		if typeHasFlags(obj.Type(), types.IsUntyped) {
			// Infer the nil type.
			lhsTypes := currentLhsList(ctx)
			index := currentRhsIndex(ctx)
			T = b.GetStoredType(ctx, lhsTypes[index])
		} else {
			T = b.GetStoredType(ctx, obj.Type())
		}
		// Create the zero value of the specified type.
		op := mlir.GoCreateZeroOperation(b.ctx, T, location)
		appendOperation(ctx, op)
		return resultsOf(op)
	default:
		value := b.valueOf(ctx, expr)
		if value == nil {
			panic("value is nil")
		}

		// Load the value
		return []mlir.Value{value.Load(ctx, location)}
	}
}

func (b *Builder) emitIndexExpr(ctx context.Context, expr *ast.IndexExpr) []mlir.Value {
	var resultType mlir.Type

	// Handle various result type scenarios.
	switch T := b.typeOf(ctx, expr).(type) {
	case *types.Tuple:
		// The result type of the index operation is that of the first member of the tuple.
		resultType = b.GetStoredType(ctx, T.At(0).Type())
	default:
		resultType = b.GetStoredType(ctx, T)
	}

	location := b.location(expr.Pos())

	// Perform the specific index operation based on the input value type.
	switch b.typeOf(ctx, expr.X).(type) {
	case *types.Array:
		// Evaluate the address.
		addr := b.emitIndexAddr(ctx, expr)

		// Load the value at the resulting address and return it.
		loadOp := mlir.GoCreateLoadOperation(b.ctx, addr, resultType, location)
		appendOperation(ctx, loadOp)
		return resultsOf(loadOp)
	case *types.Basic:
		// Evaluate the address.
		addr := b.emitIndexAddr(ctx, expr)

		// Load the byte value at the address and return the result.
		loadOp := mlir.GoCreateLoadOperation(b.ctx, addr, resultType, location)
		appendOperation(ctx, loadOp)
		return resultsOf(loadOp)
	case *types.Pointer:
		// Evaluate the address.
		addr := b.emitIndexAddr(ctx, expr)

		// Load the value at the resulting address and return it.
		loadOp := mlir.GoCreateLoadOperation(b.ctx, addr, resultType, location)
		appendOperation(ctx, loadOp)
		return resultsOf(loadOp)
	case *types.Slice:
		// Evaluate the address.
		addr := b.emitIndexAddr(ctx, expr)

		// Load the slice element value at the address and return the result.
		loadOp := mlir.GoCreateLoadOperation(b.ctx, addr, resultType, location)
		appendOperation(ctx, loadOp)
		return resultsOf(loadOp)
	case *types.Map:
		// Evaluate the map value.
		X := b.emitExpr(ctx, expr.X)[0]

		// Evaluate the index value.
		index := b.emitExpr(ctx, expr.Index)[0]

		// Perform the map lookup.
		// TODO: Properly detect if the `ok` value is present.
		lookupOp := mlir.GoCreateMapLookupOperation(b.ctx, resultType, X, index, true, location)
		appendOperation(ctx, lookupOp)
		return resultsOf(lookupOp)
	case *types.TypeParam:
		panic("unimplemented")
	default:
		panic("unhandled")
	}
}

func (b *Builder) emitIndexAddr(ctx context.Context, expr *ast.IndexExpr) mlir.Value {
	location := b.location(expr.Pos())

	// Handle various result type scenarios.
	var resultType mlir.Type
	switch T := b.typeOf(ctx, expr).(type) {
	case *types.Tuple:
		// The result type of the index operation is that of the first member of the tuple.
		resultType = b.GetStoredType(ctx, T.At(0).Type())
	default:
		resultType = b.GetStoredType(ctx, T)
	}

	pointerT := mlir.GoCreatePointerType(resultType)

	// Evaluate the index value.
	index := b.emitExpr(ctx, expr.Index)[0]

	// Perform the specific index operation based on the input value type.
	switch baseType := b.typeOf(ctx, expr.X).(type) {
	case *types.Array:
		arrayT := b.GetType(ctx, baseType)

		// Get the address of the array.
		ptr := b.addressOf(ctx, expr.X, location)

		// GEP to the address of the element at the specified index.
		gepOp := mlir.GoCreateGepOperation2(b.ctx, ptr, arrayT, []any{0, index}, pointerT, location)
		appendOperation(ctx, gepOp)
		return resultOf(gepOp)
	case *types.Basic:
		// This is a string.
		X := b.emitExpr(ctx, expr.X)[0]

		// Reinterpret the string value as its runtime representation.
		X = b.bitcastTo(ctx, X, b._string, location)

		// Get the address of the byte.
		callOp := mlir.GoCreateRuntimeCallOperation(b.ctx, "runtime.stringIndexAddr",
			[]mlir.Type{b.ptr}, []mlir.Value{X, index}, location)
		appendOperation(ctx, callOp)
		return resultOf(callOp)
	case *types.Pointer:
		// This is a pointer to an array.
		X := b.emitExpr(ctx, expr.X)[0]

		// GEP into the array at the address
		gepOp := mlir.GoCreateGepOperation2(b.ctx, X, resultType, []any{index}, pointerT, location)
		appendOperation(ctx, gepOp)
		return resultOf(gepOp)
	case *types.Slice:
		X := b.emitExpr(ctx, expr.X)[0]
		addrOp := mlir.GoCreateSliceAddrOperation(b.ctx, pointerT, X, index, location)
		appendOperation(ctx, addrOp)
		return resultOf(addrOp)
	default:
		panic("attempting to index non-addressable value")
	}
}

func (b *Builder) emitReturn(ctx context.Context, stmt *ast.ReturnStmt) {
	var results []mlir.Value
	location := b.location(stmt.Pos())
	info := currentInfo(ctx)

	// Get the current function declaration being built.
	state := currentFuncData(ctx)

	if stmt.Results == nil {
		// Load the named results (if any) and return them.
		if state.funcType.Results != nil {
			for _, field := range state.funcType.Results.List {
				for _, name := range field.Names {
					v := b.emitExpr(ctx, name)
					results = append(results, v...)
				}
			}
		}
	} else {
		// Collect the return values.
		returnTypes := make([]types.Type, state.signature.Results().Len())
		for i := range returnTypes {
			returnTypes[i] = state.signature.Results().At(i).Type()
		}

		// Support type inference for nils.
		ctx = newContextWithLhsList(ctx, returnTypes)

		for i, result := range stmt.Results {
			// Select the inferred type for nils.
			ctx = newContextWithRhsIndex(ctx, i)

			v := b.emitExpr(ctx, result)
			valueType := b.typeOf(ctx, result)

			// NOTE: The following might not convert correctly in the scenario where the expression above yields
			//       multiple interface values.
			for ii := range v {
				// If the LHS and RHS are both interfaces, the RHS interface MUST implement the LHS interface if they are
				// NOT the same interface type.
				// NOTE: index `i` is intentionally used below for determining the final result type because 2 or more
				//       return values can share the same type token.
				returnType := returnTypes[i]
				switch baseType(returnType).(type) {
				case *types.Interface:
					if !isNil(valueType) && !types.Identical(valueType, returnType) {
						if types.IsInterface(baseType(valueType)) {
							// Convert from interface A to interface B.
							v[ii] = b.emitChangeType(ctx, returnType, v[ii], location)
						} else {
							// Generate methods for named types.
							if T, ok := valueType.(*types.Named); ok {
								b.queueNamedTypeJobs(ctx, T)
							}

							// Create an interface value from the value expression.
							v[ii] = b.emitInterfaceValue(ctx, returnType, valueType, v[ii], location)
						}
					}
				case *types.Signature:
					if selExpr, ok := result.(*ast.SelectorExpr); ok {
						if sel, ok := info.Selections[selExpr]; ok {
							// Member variables would've been store as the func struct type.
							if _, ok := sel.Obj().(*types.Var); ok {
								break
							}
						}
					}
					v[ii] = b.createFunctionValue(ctx, v[ii], nil, location)
				}
			}

			if len(v) > state.signature.Results().Len() {
				/// NOTE: Some expressions may yield more results than the return specifies. Slice the returns in order
				///       to return the exact values expected by this return statement.
				results = append(results, v[:len(stmt.Results)]...)
			} else {
				results = append(results, v...)
			}
		}
	}

	// Create the return operation in the current block.
	op := mlir.GoCreateReturnOperation(b.config.Ctx, results, location)
	appendOperation(ctx, op)
}

func (b *Builder) emitLabeledStatement(ctx context.Context, stmt *ast.LabeledStmt) {
	// All labeled blocks should have been created prior. Simply just branch to it.
	block := currentLabeledBlocks(ctx)[stmt.Label.Name]
	brOp := mlir.GoCreateBranchOperation(b.ctx, block, nil, b.location(stmt.Pos()))
	appendOperation(ctx, brOp)

	// Continue emission in the labeled block.
	appendBlock(ctx, block)
	setCurrentBlock(ctx, block)

	// Emit the labeled statement's statement
	b.emitStmt(ctx, stmt.Stmt)
}

func (b *Builder) emitSelectorExpr(ctx context.Context, expr *ast.SelectorExpr) []mlir.Value {
	location := b.location(expr.Pos())
	info := currentInfo(ctx)
	sel := info.Selections[expr]
	if sel == nil {
		// This is actually a qualified identifier.
		switch obj := b.objectOf(ctx, expr.Sel).(type) {
		case *types.Func:
			T := b.GetType(ctx, obj.Type())
			symbol := qualifiedFuncName(obj)
			return b.values(b.addressOfSymbol(ctx, symbol, T, location))
		default:
			value := b.valueOf(ctx, expr.Sel)
			return b.values(value.Load(ctx, location))
		}
	}

	switch recvType := baseType(sel.Recv()).(type) {
	case *types.Interface:
		signature := sel.Type().(*types.Signature)

		// Collect argument types.
		var argTypes []mlir.Type
		for i := 0; i < signature.Params().Len(); i++ {
			argTypes = append(argTypes, b.GetStoredType(ctx, signature.Params().At(i).Type()))
		}

		// Evaluate the interface value.
		ifaceValue := b.emitExpr(ctx, expr.X)

		// Create the argument pack.
		argsValue, argsType := b.createArgumentPack(ctx, ifaceValue, location)

		// Allocate heap to store the argument pack.
		allocOp := mlir.GoCreateAllocaOperation(b.ctx, mlir.GoCreatePointerType(argsType), argsType, nil, true, location)
		appendOperation(ctx, allocOp)

		// Store the argument pack value at the heap address.
		storeOp := mlir.GoCreateStoreOperation(b.ctx, argsValue, resultOf(allocOp), location)
		appendOperation(ctx, storeOp)
		argsValue = resultOf(allocOp)

		// Format the wrapper function symbol name.
		wrapperSymbol := fmt.Sprintf("%s.%s$wrapper$2", sel.Obj().Id(), expr.Sel.Name)

		// Create an interface call wrapper.
		b.createInterfaceCallWrapper(ctx, wrapperSymbol, expr.Sel.Name, recvType, signature, argTypes)

		// Get the address of the thunk.
		wrapperAddr := b.addressOfSymbol(ctx, wrapperSymbol, b.ptr, b._noLoc)

		// Create the function value.
		return []mlir.Value{b.createFunctionValue(ctx, wrapperAddr, argsValue, location)}
	default:
		switch obj := sel.Obj().(type) {
		case *types.Func:
			// Return the address of the selected method.
			symbol := b.resolveSymbol(qualifiedFuncName(obj))
			b.queueJob(ctx, symbol)
			T := b.GetType(ctx, obj.Type())
			return []mlir.Value{
				b.addressOfSymbol(ctx, symbol, T, location),
			}
		case *types.Var:
			// Evaluate the address of the selected member.
			baseAddr := b.emitSelectAddr(ctx, expr)

			// Load the member value.
			loadOp := mlir.GoCreateLoadOperation(b.ctx, baseAddr, b.GetStoredType(ctx, b.typeOf(ctx, expr)), location)
			appendOperation(ctx, loadOp)
			return resultsOf(loadOp)
		default:
			panic("unhandled")
		}
	}
}

func (b *Builder) emitSelectAddr(ctx context.Context, expr *ast.SelectorExpr) mlir.Value {
	info := currentInfo(ctx)
	location := b.location(expr.Pos())
	selectedType := b.typeOf(ctx, expr)

	// Handle declared functions separately.
	if _, isFunc := selectedType.(*types.Signature); isFunc {
		if funcObj, ok := info.ObjectOf(expr.Sel).(*types.Func); ok {
			symbol := b.resolveSymbol(qualifiedFuncName(funcObj))
			b.queueJob(ctx, symbol)
			return b.addressOfSymbol(ctx, symbol, b.GetType(ctx, funcObj.Type()), location)
		}
	}

	var basePtr mlir.Value

	// Handle acquiring the address of some constant.
	// TODO: valueOf could probably handle all cases here, thus eliminating the nil check below.
	if obj, ok := info.Uses[expr.Sel]; ok {
		if constObj, ok := obj.(*types.Const); ok {
			val := b.emitConstantValue(ctx, constObj.Val(), constObj.Type(), location)
			basePtr = b.makeCopyOf(ctx, val, location)
		}
	}

	if basePtr == nil {
		// Get the base pointer to begin pointer arithmetic on.
		basePtr = b.baseAddressOf(ctx, expr, location)
	}

	if selection, ok := info.Selections[expr]; ok {
		// Emit GEP operations to derive the address of the selected identifier starting with the receiver type as
		// specified by the type checker.
		currentType := selection.Recv()
		for _, index := range selection.Index() {
			if isPointer(currentType) {
				// Load the pointer value.
				ptrType := b.GetType(ctx, currentType)
				loadOp := mlir.GoCreateLoadOperation(b.ctx, basePtr, ptrType, location)
				appendOperation(ctx, loadOp)
				basePtr = resultOf(loadOp)
				currentType = currentType.(*types.Pointer).Elem()
			}

			// Get the field type currently selected.
			structType := baseStructTypeOf(currentType)
			fieldType := structType.Field(index).Type()
			if _, isSignature := fieldType.(*types.Signature); isSignature {
				// Functions are stored as the func struct type.
				fieldType = b.config.Program.LookupType("runtime", "_func")
			}
			fieldPtrType := b.pointerOf(ctx, fieldType)

			// GEP to the struct field at the specified index.
			gepOp := mlir.GoCreateGepOperation2(b.ctx, basePtr, b.GetType(ctx, structType), []any{0, index}, fieldPtrType, location)
			appendOperation(ctx, gepOp)
			basePtr = resultOf(gepOp)

			// Update the current type.
			currentType = fieldType.Underlying()
		}
	}

	// Return the resulting pointer.
	return basePtr
}

func (b *Builder) baseAddressOf(ctx context.Context, expr *ast.SelectorExpr, location mlir.Location) mlir.Value {
	info := currentInfo(ctx)
	switch X := expr.X.(type) {
	case *ast.Ident:
		baseObj := info.ObjectOf(X)
		switch baseObj := baseObj.(type) {
		case *types.Var:
			return b.valueOf(ctx, X).Pointer(ctx, location)
		case *types.PkgName:
			symbol := baseObj.Imported().Path() + "." + expr.Sel.Name
			globalT := b.typeOf(ctx, expr)
			addressOfOp := mlir.GoCreateAddressOfOperation(b.ctx, symbol, b.pointerOf(ctx, globalT), location)
			appendOperation(ctx, addressOfOp)
			return resultOf(addressOfOp)
		default:
			panic("unhandled")
		}
	case *ast.SelectorExpr:
		return b.emitSelectAddr(ctx, X)
	default:
		// Store the value on the stack and return the stack address.
		return b.addressOf(ctx, X, location)
	}
}

func (b *Builder) emitSliceExpr(ctx context.Context, expr *ast.SliceExpr) []mlir.Value {
	var lowValue, highValue, maxValue mlir.Value
	location := b.location(expr.Pos())
	T := b.GetStoredType(ctx, b.typeOf(ctx, expr))

	// Evaluate the input slice.
	X := b.emitExpr(ctx, expr.X)[0]

	// Evaluate each available index.
	if expr.Low != nil {
		lowValue = b.emitExpr(ctx, expr.Low)[0]
	}

	if expr.High != nil {
		highValue = b.emitExpr(ctx, expr.High)[0]
	}

	if expr.Max != nil {
		maxValue = b.emitExpr(ctx, expr.Max)[0]
	}

	// Emit the slice operation.
	sliceOp := mlir.GoCreateSliceOperation(b.ctx, X, lowValue, highValue, maxValue, T, location)
	appendOperation(ctx, sliceOp)
	return resultsOf(sliceOp)
}

func (b *Builder) emitStarExpr(ctx context.Context, expr *ast.StarExpr) []mlir.Value {
	elementType := b.GetStoredType(ctx, b.typeOf(ctx, expr))
	X := b.emitExpr(ctx, expr.X)[0]

	// Load and return the value at the address.
	op := mlir.GoCreateLoadOperation(b.ctx, X, elementType, b.location(expr.Pos()))
	appendOperation(ctx, op)
	return resultsOf(op)
}

func (b *Builder) emitStmt(ctx context.Context, stmt ast.Stmt) {
	defer func() {
		if v := recover(); v != nil {
			pos := b.config.Fset.Position(stmt.Pos())
			fname, _ := filepath.EvalSymlinks(pos.Filename)
			fname = fmt.Sprintf("%s:%d:%d", fname, pos.Line, pos.Column)
			line := b.locationString(stmt.Pos())
			fmt.Fprintf(os.Stderr, "failure while emitting %T: %+v\n%s\n\n%s\n\n%s\n",
				stmt, v, fname, line, string(debug.Stack()))
			os.Exit(-1)
		}
	}()

	switch stmt := stmt.(type) {
	case *ast.AssignStmt:
		b.emitAssign(ctx, stmt)
	case *ast.BlockStmt:
		// Fill the current block.
		// NOTE: The caller is expected to construct the block to be emitted into.
		b.emitBlock(ctx, stmt)
	case *ast.BranchStmt:
		b.emitBranchStatement(ctx, stmt)
	case *ast.CaseClause:
		panic("unreachable")
	case *ast.CommClause:
		panic("unreachable")
	case *ast.DeclStmt:
		b.emitDecl(ctx, stmt.Decl)
	case *ast.DeferStmt:
		b.emitDeferStatement(ctx, stmt)
	case *ast.EmptyStmt:
		// Do nothing.
	case *ast.ExprStmt:
		b.emitExpr(ctx, stmt.X)
	case *ast.ForStmt:
		// Emit the switch statement.
		b.emitForStatement(ctx, stmt)
	case *ast.GoStmt:
		b.emitGoStatement(ctx, stmt)
	case *ast.IfStmt:
		b.emitIfStatement(ctx, stmt)
	case *ast.IncDecStmt:
		b.emitIncDecStatement(ctx, stmt)
	case *ast.LabeledStmt:
		b.emitLabeledStatement(ctx, stmt)
	case *ast.RangeStmt:
		b.emitRangeStatement(ctx, stmt)
	case *ast.ReturnStmt:
		b.emitReturn(ctx, stmt)
	case *ast.SelectStmt:
		b.emitSelectStatement(ctx, stmt)
	case *ast.SendStmt:
		b.emitSendStatement(ctx, stmt)
	case *ast.SwitchStmt:
		b.emitExpressionSwitchStatement(ctx, stmt)
	case *ast.TypeSwitchStmt:
		b.emitTypeSwitchStatement(ctx, stmt)
	default:
		panic("unhandled statement")
	}
}

func (b *Builder) emitTypeAssertExpr(ctx context.Context, expr *ast.TypeAssertExpr) []mlir.Value {
	location := b.location(expr.Pos())

	// Evaluate the interface value to type assert on.
	X := b.emitExpr(ctx, expr.X)[0]

	// Get the information of the type to assert.
	T := b.GetType(ctx, b.typeOf(ctx, expr.Type))

	// Create the type assertion operation.
	op := mlir.GoCreateTypeAssertOperation(b.ctx, X, T, location)
	appendOperation(ctx, op)
	return resultsOf(op)
}

func (b *Builder) emitUnaryExpr(ctx context.Context, expr *ast.UnaryExpr) []mlir.Value {
	location := b.location(expr.Pos())
	exprT := b.typeOf(ctx, expr)
	if !typeHasFlags(exprT, types.IsUntyped) {
		ctx = newContextWithLhsList(ctx, []types.Type{exprT})
		ctx = newContextWithRhsIndex(ctx, 0)
	}

	switch expr.Op {
	case token.ADD:
		// This basically returns the same value as its input.
		X := b.emitExpr(ctx, expr.X)[0]
		return []mlir.Value{X}
	case token.SUB:
		var op mlir.Operation
		X := b.emitExpr(ctx, expr.X)[0]
		switch {
		case b.exprTypeHasFlags(ctx, expr, types.IsInteger):
			op = mlir.GoCreateNegIOperation(b.ctx, X, location)
		case b.exprTypeHasFlags(ctx, expr, types.IsFloat):
			op = mlir.GoCreateNegFOperation(b.ctx, X, location)
		case b.exprTypeHasFlags(ctx, expr, types.IsComplex):
			op = mlir.GoCreateNegCOperation(b.ctx, X, location)
		}
		appendOperation(ctx, op)
		return []mlir.Value{resultOf(op)}
	case token.NOT:
		X := b.emitExpr(ctx, expr.X)[0]
		op := mlir.GoCreateNotOperation(b.ctx, X, location)
		appendOperation(ctx, op)
		return []mlir.Value{resultOf(op)}
	case token.XOR:
		X := b.emitExpr(ctx, expr.X)[0]
		op := mlir.GoCreateComplementOperation(b.ctx, X, location)
		appendOperation(ctx, op)
		return []mlir.Value{resultOf(op)}
	case token.MUL:
		panic("unreachable")
	case token.AND:
		T := b.typeOf(ctx, expr.X)
		if T, ok := T.(*types.Named); ok {
			b.queueNamedTypeJobs(ctx, T)
		}
		return []mlir.Value{b.addressOf(ctx, expr.X, location)}
	case token.ARROW:
		return b.emitReceiveExpression(ctx, expr)
	default:
		panic("invalid unary expression operator")
	}
}
