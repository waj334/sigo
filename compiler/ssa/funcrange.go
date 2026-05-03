package ssa

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"

	"pkg.si-go.dev/go-mlir/mlir"
	"pkg.si-go.dev/sigo/goir/binding/goir"
)

// emitFuncRange lowers `for k, v := range fn { body }` where fn is a
// `func(yield func(...) bool)`.
//
// Phase 1: dispatch + closure synthesis. Natural completion works.
// Phase 2A: unlabeled `break` (returns false from yield) and `continue`
//
//	(branches to a closure-local fallthrough block returning true).
//
// Phase 2B: `return` inside body propagates to enclosing function. A hidden
//
//	`state` int slot is allocated in the outer function and captured by the
//	closure. emitReturn under the active frame stores result values into
//	captured result-temp slots, sets state to a return-sentinel, and emits
//	`return false` from the closure. After the iter call, this function
//	tests state; if equal to the sentinel, it loads the temps and emits
//	the outer function's Return op, otherwise it falls through.
//
// Still TODO (later phases):
//   - Labeled break/continue + nested-rangefunc return propagation (frame stack).
//   - Yield-after-stop runtime panic.
func (b *Builder) emitFuncRange(ctx context.Context, stmt *ast.RangeStmt) {
	location := b.location(ctx, stmt.Pos())
	info := currentInfo(ctx)

	iterType := b.typeOf(ctx, stmt.X)
	iterSig, ok := iterType.Underlying().(*types.Signature)
	if !ok {
		panic("range over func: iterator must be a function")
	}
	if iterSig.Params().Len() != 1 {
		panic("range over func: iterator must take exactly one parameter")
	}
	yieldSig, ok := iterSig.Params().At(0).Type().Underlying().(*types.Signature)
	if !ok {
		panic("range over func: iterator parameter must be a function")
	}
	if yieldSig.Results().Len() != 1 {
		panic("range over func: yield function must return exactly one value")
	}
	if t, ok := yieldSig.Results().At(0).Type().(*types.Basic); !ok || t.Kind() != types.Bool {
		panic("range over func: yield function must return bool")
	}

	if stmt.Tok != token.DEFINE && (stmt.Key != nil || stmt.Value != nil) {
		panic("range over func: only the := form is supported")
	}

	syntheticParams := b.buildRangeFuncParams(ctx, stmt, yieldSig)

	boolIdent := &ast.Ident{NamePos: stmt.Pos(), Name: "bool"}
	info.Types[boolIdent] = types.TypeAndValue{Type: types.Typ[types.Bool]}

	syntheticType := &ast.FuncType{
		Func: stmt.Pos(),
		Params: &ast.FieldList{
			Opening: stmt.Pos(),
			List:    syntheticParams,
			Closing: stmt.Pos(),
		},
		Results: &ast.FieldList{
			List: []*ast.Field{
				{Type: boolIdent},
			},
		},
	}

	syntheticLit := &ast.FuncLit{
		Type: syntheticType,
		Body: stmt.Body,
	}

	// Build closureSig using the same *types.Var objects that the synthetic
	// FuncType.Params.List Names resolve to. emitFuncLiteral's free-variable
	// walk excludes parameters by checking against
	// `originalSignature.Params().Variables()`. If we used yieldSig.Params()
	// (whose Vars are anonymous and distinct from the user's range vars),
	// the user's range key/value would be misclassified as captures and
	// lookupValue would return nil during context-struct construction.
	var closureParamVars []*types.Var
	for i, field := range syntheticParams {
		for _, name := range field.Names {
			obj := info.ObjectOf(name)
			v, ok := obj.(*types.Var)
			if !ok || v == nil {
				v = types.NewVar(name.NamePos, nil, name.Name, yieldSig.Params().At(i).Type())
			}
			closureParamVars = append(closureParamVars, v)
		}
	}
	closureSig := types.NewSignatureType(nil, nil, nil,
		types.NewTuple(closureParamVars...), yieldSig.Results(), false)

	info.Types[syntheticLit] = types.TypeAndValue{Type: closureSig}
	info.Types[syntheticType] = types.TypeAndValue{Type: closureSig}

	enclosingScope := info.Scopes[currentFuncData(ctx).funcType]
	if enclosingScope == nil {
		enclosingScope = currentFuncData(ctx).scope
	}
	closureScope := types.NewScope(enclosingScope, stmt.Pos(), stmt.End(), "rangefunc")
	info.Scopes[syntheticType] = closureScope

	for _, field := range syntheticParams {
		for _, name := range field.Names {
			if obj := info.ObjectOf(name); obj != nil {
				closureScope.Insert(obj)
			}
		}
	}

	// Phase 2B/3a: allocate (or inherit) the hidden state slot + result
	// temps. For an OUTERMOST rangefunc within the enclosing function, we
	// allocate fresh slots in the enclosing function's frame. For a NESTED
	// rangefunc (one whose body is itself inside a closure body of an
	// outer rangefunc), we reuse the outermost frame's slots so a `return`
	// from any depth lands in the same place. The capture chain
	// transparently exposes those slots through each closure level via
	// FreeVar indirection.
	intType := b.GetStoredType(ctx, types.Typ[types.Int])
	parentFrame := currentRangeFuncFrame(ctx)

	var stateVar types.Object
	var resultTempVars []types.Object
	var outerSig *types.Signature

	if parentFrame == nil {
		// Outermost: allocate fresh slots.
		stateVar = types.NewVar(token.NoPos, nil, "$rfstate", types.Typ[types.Int])
		b.emitLocalVar(ctx, stateVar, intType, false)

		outerSig = currentFuncData(ctx).signature
		for i := 0; i < outerSig.Results().Len(); i++ {
			res := outerSig.Results().At(i)
			tempVar := types.NewVar(token.NoPos, nil, fmt.Sprintf("$rfret%d", i), res.Type())
			b.emitLocalVar(ctx, tempVar, b.GetStoredType(ctx, res.Type()), false)
			resultTempVars = append(resultTempVars, tempVar)
		}
	} else {
		// Nested: walk to the outermost frame and reuse its slots.
		root := parentFrame
		for root.parent != nil {
			root = root.parent
		}
		stateVar = root.stateVar
		resultTempVars = root.resultTempVars
		outerSig = root.outerSig
	}

	fallthroughBlock := mlir.NewBlock(nil, nil)
	const returnSentinel int64 = 1
	depth := 0
	if parentFrame != nil {
		depth = parentFrame.depth + 1
	}
	label := currentPendingRangeLabel(ctx)
	// Clear the pending label after consumption so an unlabeled inner
	// range-over-func doesn't accidentally inherit it.
	ctx = newContextWithPendingRangeLabel(ctx, "")

	// Phase 8: per-frame "stopped" flag for the yield-after-stop runtime
	// check. Allocated in the same scope as state/temps. Default zero
	// (false). Set to true before any return-false from the closure.
	stoppedVar := types.NewVar(token.NoPos, nil, "$rfstopped", types.Typ[types.Bool])
	b.emitLocalVar(ctx, stoppedVar, b.GetStoredType(ctx, types.Typ[types.Bool]), false)

	frame := &rangeFuncFrame{
		parent:           parentFrame,
		fallthroughBlock: fallthroughBlock,
		stateVar:         stateVar,
		returnSentinel:   returnSentinel,
		resultTempVars:   resultTempVars,
		outerSig:         outerSig,
		label:            label,
		depth:            depth,
		stoppedVar:       stoppedVar,
	}

	// Register body hooks via the side-table. emitCallExpr below will invoke
	// emitFuncLiteral on the synthetic FuncLit indirectly via emitCallArgs,
	// and that path does not propagate setup arguments. The side-table lets
	// emitFuncLiteral pick up the hooks regardless of how it was reached.
	hook := func(anonData *funcData) {
		// Phase 2B: capture state + result temps so the closure body can
		// reach them via lookupValue. addAnonCapture must run before
		// createContextStructValue (lit.go arranges that ordering).
		b.addAnonCapture(ctx, anonData, stateVar)
		for _, tv := range resultTempVars {
			b.addAnonCapture(ctx, anonData, tv)
		}
		// Phase 8: capture the per-frame stopped flag.
		b.addAnonCapture(ctx, anonData, stoppedVar)

		anonData.bodyContextHook = func(ctx context.Context) context.Context {
			// Phase 8: emit a "yield after stop" panic check at the top of
			// the closure body. If stoppedVar is true, the iterator has
			// violated the protocol — call runtime.panicYieldAfterStop.
			stoppedLV := b.lookupValue(ctx, stoppedVar)
			stoppedValue := stoppedLV.Load(ctx, location)

			runBlock := mlir.NewBlock(nil, nil)
			panicBlock := mlir.NewBlock(nil, nil)
			appendBlock(ctx, runBlock)
			appendBlock(ctx, panicBlock)

			condBr := goir.NewCondBranchOperation(b.ctx, stoppedValue, panicBlock, nil, runBlock, nil, location)
			appendOperation(ctx, condBr)

			buildBlock(ctx, panicBlock, func() {
				callOp := goir.NewCallOperation(b.ctx, "runtime.panicYieldAfterStop", nil, nil, location)
				appendOperation(ctx, callOp)
				// runtime.panicYieldAfterStop is no-return; emit a
				// `return false` as a terminator since MLIR requires one.
				falseValue := b.emitConstBool(ctx, false, b.i1, location)
				retOp := goir.NewReturnOperation(b.config.Ctx, []mlir.ValueLike{falseValue}, location)
				appendOperation(ctx, retOp)
			})

			setCurrentBlock(ctx, runBlock)

			appendBlock(ctx, fallthroughBlock)
			buildBlock(ctx, fallthroughBlock, func() {
				trueValue := b.emitConstBool(ctx, true, b.i1, location)
				retOp := goir.NewReturnOperation(b.config.Ctx, []mlir.ValueLike{trueValue}, location)
				appendOperation(ctx, retOp)
			})
			return newContextWithRangeFuncFrame(ctx, frame)
		}
		anonData.bodyTailHook = func(ctx context.Context) {
			if !blockHasTerminator(currentBlock(ctx)) {
				brOp := goir.NewBranchOperation(b.ctx, fallthroughBlock, nil, location)
				appendOperation(ctx, brOp)
			}
		}
	}
	b.funcLitHooksMutex.Lock()
	if b.funcLitHooks == nil {
		b.funcLitHooks = map[*ast.FuncLit]func(*funcData){}
	}
	b.funcLitHooks[syntheticLit] = hook
	b.funcLitHooksMutex.Unlock()

	syntheticCall := &ast.CallExpr{
		Fun:    stmt.X,
		Lparen: stmt.Pos(),
		Args:   []ast.Expr{syntheticLit},
		Rparen: stmt.End(),
	}
	info.Types[syntheticCall] = types.TypeAndValue{Type: types.NewTuple()}

	b.emitCallExpr(ctx, syntheticCall)

	b.emitRangeFuncPostIter(ctx, frame, intType, location)
}

// emitRangeFuncPostIter emits the dispatch run after a range-over-func
// iter call returns. It decodes the shared state slot and either falls
// through (no flag), emits the outermost function's Return op, propagates
// by returning false from the current closure, or resolves a labeled
// break/continue whose target is the parent of this frame.
//
// State encoding (see context.go for the wider picture):
//
//	0                 — no flag (natural completion or unlabeled break)
//	1                 — return: outermost emits Return; non-outermost propagates
//	2 + 2*k           — labeled break of frame at depth k
//	3 + 2*k           — labeled continue of frame at depth k
//
// For a frame at depth d >= 1, the post-iter dispatch lives in the parent
// closure's body. "Resolution" happens when the encoded target depth k
// equals d-1 (the parent of this frame, which is the closure we're
// emitting in): the state is cleared and the current closure returns
// true (continue) or false (break) to its iter. Other state values are
// propagated by emitting `return false` without clearing.
func (b *Builder) emitRangeFuncPostIter(
	ctx context.Context,
	frame *rangeFuncFrame,
	intType mlir.TypeLike,
	location mlir.LocationLike,
) {
	d := frame.depth
	stateLV := b.lookupValue(ctx, frame.stateVar)
	stateValue := stateLV.Load(ctx, location)

	zeroValue := b.emitConstInt(ctx, 0, intType, location)
	returnSentinelV := b.emitConstInt(ctx, frame.returnSentinel, intType, location)

	continueBlock := mlir.NewBlock(nil, nil)
	returnBlock := mlir.NewBlock(nil, nil)

	// state == 0 ?
	cmpZero := goir.NewCmpIOperation(b.ctx, b.i1, b.cmpIPredicate(token.EQL, false), stateValue, zeroValue, location)
	appendOperation(ctx, cmpZero)
	afterZeroBlock := mlir.NewBlock(nil, nil)
	appendBlock(ctx, afterZeroBlock)
	br0 := goir.NewCondBranchOperation(b.ctx, resultOf(cmpZero), continueBlock, nil, afterZeroBlock, nil, location)
	appendOperation(ctx, br0)

	setCurrentBlock(ctx, afterZeroBlock)

	// state == 1 (return) ?
	cmpRet := goir.NewCmpIOperation(b.ctx, b.i1, b.cmpIPredicate(token.EQL, false), stateValue, returnSentinelV, location)
	appendOperation(ctx, cmpRet)

	if d == 0 {
		// Outermost frame: only state==1 matters here. Anything else
		// would be an out-of-scope label encoding (we shouldn't see those
		// at the outermost dispatch since labeled break/continue with
		// target inside outermost is resolved by inner frames). Fall
		// through to continueBlock for safety.
		appendBlock(ctx, returnBlock)
		brR := goir.NewCondBranchOperation(b.ctx, resultOf(cmpRet), returnBlock, nil, continueBlock, nil, location)
		appendOperation(ctx, brR)

		buildBlock(ctx, returnBlock, func() {
			results := make([]mlir.ValueLike, 0, len(frame.resultTempVars))
			for _, tv := range frame.resultTempVars {
				lv := b.lookupValue(ctx, tv)
				results = append(results, lv.Load(ctx, location))
			}
			retOp := goir.NewReturnOperation(b.config.Ctx, results, location)
			appendOperation(ctx, retOp)
		})
	} else {
		// Nested: state==1 propagates. Then check break-of-parent and
		// continue-of-parent. Anything else: propagate (return false).
		propagateBlock := mlir.NewBlock(nil, nil)
		breakResolveBlock := mlir.NewBlock(nil, nil)
		continueResolveBlock := mlir.NewBlock(nil, nil)
		afterRetBlock := mlir.NewBlock(nil, nil)
		afterBreakBlock := mlir.NewBlock(nil, nil)

		appendBlock(ctx, returnBlock)
		appendBlock(ctx, afterRetBlock)
		appendBlock(ctx, breakResolveBlock)
		appendBlock(ctx, afterBreakBlock)
		appendBlock(ctx, continueResolveBlock)
		appendBlock(ctx, propagateBlock)

		// Branch on state == 1.
		brR := goir.NewCondBranchOperation(b.ctx, resultOf(cmpRet), returnBlock, nil, afterRetBlock, nil, location)
		appendOperation(ctx, brR)

		// returnBlock: propagate (return false) preserving state.
		buildBlock(ctx, returnBlock, func() {
			if frame.parent != nil {
				b.markRangeFuncStopped(ctx, frame.parent, location)
			}
			falseV := b.emitConstBool(ctx, false, b.i1, location)
			retOp := goir.NewReturnOperation(b.config.Ctx, []mlir.ValueLike{falseV}, location)
			appendOperation(ctx, retOp)
		})

		// afterRetBlock: state == break-of-parent ?
		buildBlock(ctx, afterRetBlock, func() {
			parentBreakSentinel := int64(2 + 2*(d-1))
			breakSV := b.emitConstInt(ctx, parentBreakSentinel, intType, location)
			cmpBreak := goir.NewCmpIOperation(b.ctx, b.i1, b.cmpIPredicate(token.EQL, false), stateValue, breakSV, location)
			appendOperation(ctx, cmpBreak)
			brB := goir.NewCondBranchOperation(b.ctx, resultOf(cmpBreak), breakResolveBlock, nil, afterBreakBlock, nil, location)
			appendOperation(ctx, brB)
		})

		// breakResolveBlock: clear state, return false (current closure stops).
		buildBlock(ctx, breakResolveBlock, func() {
			stateLV.Store(ctx, zeroValue, location)
			if frame.parent != nil {
				b.markRangeFuncStopped(ctx, frame.parent, location)
			}
			falseV := b.emitConstBool(ctx, false, b.i1, location)
			retOp := goir.NewReturnOperation(b.config.Ctx, []mlir.ValueLike{falseV}, location)
			appendOperation(ctx, retOp)
		})

		// afterBreakBlock: state == continue-of-parent ?
		buildBlock(ctx, afterBreakBlock, func() {
			parentContinueSentinel := int64(3 + 2*(d-1))
			contSV := b.emitConstInt(ctx, parentContinueSentinel, intType, location)
			cmpCont := goir.NewCmpIOperation(b.ctx, b.i1, b.cmpIPredicate(token.EQL, false), stateValue, contSV, location)
			appendOperation(ctx, cmpCont)
			brC := goir.NewCondBranchOperation(b.ctx, resultOf(cmpCont), continueResolveBlock, nil, propagateBlock, nil, location)
			appendOperation(ctx, brC)
		})

		// continueResolveBlock: clear state, return true (current closure resumes its iter).
		buildBlock(ctx, continueResolveBlock, func() {
			stateLV.Store(ctx, zeroValue, location)
			trueV := b.emitConstBool(ctx, true, b.i1, location)
			retOp := goir.NewReturnOperation(b.config.Ctx, []mlir.ValueLike{trueV}, location)
			appendOperation(ctx, retOp)
		})

		// propagateBlock: return false; preserve state for ancestor's dispatch.
		buildBlock(ctx, propagateBlock, func() {
			if frame.parent != nil {
				b.markRangeFuncStopped(ctx, frame.parent, location)
			}
			falseV := b.emitConstBool(ctx, false, b.i1, location)
			retOp := goir.NewReturnOperation(b.config.Ctx, []mlir.ValueLike{falseV}, location)
			appendOperation(ctx, retOp)
		})
	}

	appendBlock(ctx, continueBlock)
	setCurrentBlock(ctx, continueBlock)
}

// buildRangeFuncParams constructs synthetic *ast.Field entries for the yield
// closure's parameters. Where possible, the parameter ident is reused from the
// range statement so emitFunc binds incoming arguments directly to those
// variables. Missing or blank idents are replaced with fresh synthetic vars.
func (b *Builder) buildRangeFuncParams(ctx context.Context, stmt *ast.RangeStmt, yieldSig *types.Signature) []*ast.Field {
	info := currentInfo(ctx)
	var fields []*ast.Field

	resolve := func(rangeExpr ast.Expr, paramVar *types.Var, slot int) *ast.Ident {
		var ident *ast.Ident
		if rangeExpr != nil {
			if id, ok := rangeExpr.(*ast.Ident); ok && id.Name != "_" {
				ident = id
			}
		}
		if ident == nil {
			pos := stmt.Pos()
			if rangeExpr != nil {
				pos = rangeExpr.Pos()
			}
			synth := &ast.Ident{NamePos: pos, Name: rangefuncSynthName(slot)}
			info.Defs[synth] = types.NewVar(pos, nil, synth.Name, paramVar.Type())
			ident = synth
		}
		return ident
	}

	addField := func(ident *ast.Ident, paramVar *types.Var) {
		typeIdent := &ast.Ident{NamePos: ident.NamePos, Name: paramVar.Type().String()}
		info.Types[typeIdent] = types.TypeAndValue{Type: paramVar.Type()}
		fields = append(fields, &ast.Field{
			Names: []*ast.Ident{ident},
			Type:  typeIdent,
		})
	}

	switch yieldSig.Params().Len() {
	case 0:
		// No iteration variables.
	case 1:
		ident := resolve(stmt.Key, yieldSig.Params().At(0), 0)
		addField(ident, yieldSig.Params().At(0))
	case 2:
		keyIdent := resolve(stmt.Key, yieldSig.Params().At(0), 0)
		valIdent := resolve(stmt.Value, yieldSig.Params().At(1), 1)
		addField(keyIdent, yieldSig.Params().At(0))
		addField(valIdent, yieldSig.Params().At(1))
	default:
		panic("range over func: yield function must take 0, 1, or 2 arguments")
	}
	return fields
}

func rangefuncSynthName(slot int) string {
	switch slot {
	case 0:
		return "_rfk"
	case 1:
		return "_rfv"
	}
	return "_rfx"
}
