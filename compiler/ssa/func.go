package ssa

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"sync"

	"pkg.si-go.dev/go-mlir/mlir"
	"pkg.si-go.dev/sigo/goir/binding/goir"
)

type funcData struct {
	symbol      string
	linkname    string
	scope       *types.Scope
	funcType    *ast.FuncType
	mlirType    goir.FunctionType
	signature   *types.Signature
	freeVars    []*FreeVar
	contextType mlir.TypeLike
	recv        *ast.FieldList
	body        *ast.BlockStmt
	pos         token.Pos
	isGeneric   bool
	isExported  bool
	isInstance  bool
	isAnonymous bool
	linkage     string

	isPackageInit bool
	priority      int

	mutex    sync.RWMutex
	instance int

	locals         map[types.Object]Value
	anonymousFuncs map[*ast.FuncLit]*funcData
	instances      []*funcData
	typeMap        TypeParamMap

	decl *ast.FuncDecl
	info *types.Info

	attributes []string
}

type inputParam struct {
	t mlir.TypeLike
	l mlir.LocationLike
}

type inputParams []inputParam

func (i inputParams) types() []mlir.TypeLike {
	result := make([]mlir.TypeLike, 0, len(i))
	for _, i := range i {
		result = append(result, i.t)
	}
	return result
}

func (i inputParams) locations() []mlir.LocationLike {
	result := make([]mlir.LocationLike, 0, len(i))
	for _, i := range i {
		result = append(result, i.l)
	}
	return result
}

func (f *funcData) createContextStructValue(ctx context.Context, b *Builder, location mlir.LocationLike) (mlir.ValueLike, mlir.TypeLike, mlir.TypeLike) {
	// Collect the addresses of each value captured by this function.
	var values []mlir.ValueLike
	var typs []types.Type
	for _, fv := range f.freeVars {
		ptr := b.lookupValue(ctx, fv.obj).Pointer(ctx, location)
		values = append(values, ptr)
		typs = append(typs, types.NewPointer(fv.GoT))
	}
	return b.createArgumentPack(ctx, values, typs, location)
}

func (b *Builder) emitFunc(ctx context.Context, data *funcData) {
	if data.isGeneric {
		// Do not attempt to generate uninstantiated generic functions.
		return
	}

	isForwardDeclaration := data.body == nil

	var queue *jobQueue
	if val := ctx.Value(jobQueueKey{}); val != nil {
		queue = val.(*jobQueue)
	}

	// Set the current data in a fresh context.
	ctx = newContextWithFuncData(context.Background(), data)
	ctx = newContextWithInfo(ctx, data.info)
	ctx = newContextWithTypeMap(ctx, data.typeMap)

	if queue != nil {
		ctx = context.WithValue(ctx, jobQueueKey{}, queue)
	}

	// Get the location of the input function.
	loc := b.location(ctx, data.pos)

	// Fuse the location with the compile unit if applicable.
	if data.pos.IsValid() {
		if file := b.config.Fset.File(data.pos); file != nil {
			if compileUnitAttr, ok := b.compileUnits[file]; ok {
				loc = mlir.NewFusedLoc(b.ctx, []mlir.LocationLike{loc}, compileUnitAttr)
			}
		}
	}

	// Create the function operation.
	state := mlir.NewOperationState("go.func", loc)

	argOffset := 0

	// Determine this number of inputs required to call this function.
	numInputs := data.mlirType.NumInputs()
	if data.signature.Recv() != nil {
		numInputs++
	}
	inputs := make(inputParams, numInputs)

	// Collect input information.
	if data.signature.Recv() != nil {
		// The receiver is the first parameter to this function. So, offset by 1.
		argOffset = 1
		inputs[0].t = data.mlirType.Receiver()
		inputs[0].l = b.location(ctx, data.signature.Recv().Pos())
	}

	for i := 0; i < data.signature.Params().Len(); i++ {
		param := data.signature.Params().At(i)
		inputs[argOffset+i].t = data.mlirType.Input(i)
		inputs[argOffset+i].l = b.location(ctx, param.Pos())
	}

	if data.isAnonymous {
		// Don't emit a local variable for the context pointer.
		argOffset = 1
	}

	// Create the region in which all blocks will be placed in.
	region := mlir.NewRegion()
	ctx = newContextWithRegion(ctx, region)
	state.AddOwnedRegions(region)

	// NOTE: Forward declarations will not have any block.
	if !isForwardDeclaration {
		// Create the entry block for the current function.
		entryBlock := mlir.NewBlock(inputs.types(), inputs.locations())
		region.AppendOwnedBlock(entryBlock)

		ctx = newContextWithCurrentBlock(ctx)
		setCurrentBlock(ctx, entryBlock)

		// NOTE: The captures are part of the types.Signature object for this function.
		data.mutex.RLock()
		if len(data.freeVars) > 0 {
			// Update free variable pointers.
			ctxValue := entryBlock.Argument(0)
			for i, fv := range data.freeVars {
				// Append the freevar's alloca operation to the current block.
				resultVal, _ := fv.ptr.AsResult()
				allocaOp := resultVal.OwningOperation()
				goir.AllocaOperationSetName(allocaOp, fv.obj.Name())
				appendOperation(ctx, allocaOp)

				ptrType := b.GetStoredType(ctx, types.NewPointer(fv.GoT))
				refType := b.GetStoredType(ctx, types.NewPointer(types.NewPointer(fv.GoT)))

				// GEP into the context to derive the address of the free variable.
				gepOp := goir.NewGepOperation(b.ctx,
					ctxValue, data.contextType, []int{0, i}, nil, []bool{false, false}, refType, loc)
				appendOperation(ctx, gepOp)

				// Load the address of the external local variable.
				addr := b.emitLoad(ctx, resultOf(gepOp), ptrType, loc)

				// Store the address of the free variable at the address of the stack allocation.
				b.emitStore(ctx, addr, fv.ptr, loc)
			}

			// Function arguments start after the capture list.
			argOffset = 1
		}
		data.mutex.RUnlock()

		if data.recv != nil {
			for _, field := range data.recv.List {
				for _, name := range field.Names {
					recvVar := b.objectOf(ctx, name)
					recvVal := entryBlock.Argument(0)

					// Emit a local variable allocation to hold the argument value.
					if recvVal.Type().IsNull() {
						println("STOP")
					}

					addr := b.emitLocalVar(ctx, recvVar, recvVal.Type(), true)

					// Store the parameter value at the address.
					addr.Store(ctx, recvVal, loc)
				}
			}
		}

		// Handle function parameter values.
		if data.signature.Params().Len() > 0 {
			arg := argOffset
			for _, field := range data.funcType.Params.List {
				for _, name := range field.Names {
					argVar := b.objectOf(ctx, name)
					argVal := entryBlock.Argument(arg)
					arg++

					// Emit a local variable allocation to hold the argument value.
					addr := b.emitLocalVar(ctx, argVar, b.GetStoredType(ctx, argVar.Type()), true)

					// Store the parameter value at the address.
					addr.Store(ctx, argVal, loc)
				}
			}
		}

		// Handle named results.
		if data.funcType.Results != nil {
			result := 0
			for _, field := range data.funcType.Results.List {
				for _, name := range field.Names {
					resultVar := b.objectOf(ctx, name)
					result++
					b.emitLocalVar(ctx, resultVar, b.GetStoredType(ctx, resultVar.Type()), false)
				}
			}
		}

		// Create all labeled blocks before emitting the body since some statements might need to be able to look them
		// up later. The ast.LabeledStmt will actually append them to the current region at the appropriate time.
		labeledBlocks := map[string]mlir.Block{}
		ast.Inspect(data.body, func(node ast.Node) bool {
			if stmt, ok := node.(*ast.LabeledStmt); ok {
				labeledBlock := mlir.NewBlock(nil, nil)
				labeledBlocks[stmt.Label.Name] = labeledBlock

				// Append the block now.
				appendBlock(ctx, labeledBlock)
			}
			return true
		})
		ctx = newContextWithLabeledBlocks(ctx, labeledBlocks)

		// Fill the function body.
		b.emitBlock(ctx, data.body)

		// Assume that the current block is the "last" logical block in the function.
		lastBlock := currentBlock(ctx)

		if !blockHasTerminator(lastBlock) {
			// Control flow fell off the end of the function block. Insert tail operations.
			// TODO: Run defers.

			endLocation := b.location(ctx, data.body.End())
			zeroValues := make([]mlir.ValueLike, 0, data.signature.Results().Len())
			for i := 0; i < data.signature.Results().Len(); i++ {
				result := data.signature.Results().At(i)
				value := b.emitZeroValue(ctx, result.Type(), endLocation)
				zeroValues = append(zeroValues, value)
			}

			returnOp := goir.NewReturnOperation(b.ctx, zeroValues, endLocation)
			appendOperation(ctx, returnOp)
		}
	}

	visibility := "public"
	if !data.isExported || isForwardDeclaration {
		// NOTE: Forward declarations MUST be private.
		visibility = "private"
	}

	// Set the linkage type.
	linkage := "external"
	if len(data.linkage) > 0 {
		linkage = data.linkage
	}

	var linkageAttr mlir.LLVMLinkageAttr
	switch linkage {
	case "appending":
		linkageAttr = mlir.NewLLVMLinkageAttr(b.ctx, mlir.LLVMLinkageAppending)
	case "available_externally":
		linkageAttr = mlir.NewLLVMLinkageAttr(b.ctx, mlir.LLVMLinkageAvailableExternally)
	case "common":
		linkageAttr = mlir.NewLLVMLinkageAttr(b.ctx, mlir.LLVMLinkageCommon)
	case "external":
		linkageAttr = mlir.NewLLVMLinkageAttr(b.ctx, mlir.LLVMLinkageExternal)
	case "extern_weak":
		linkageAttr = mlir.NewLLVMLinkageAttr(b.ctx, mlir.LLVMLinkageExternWeak)
	case "internal":
		linkageAttr = mlir.NewLLVMLinkageAttr(b.ctx, mlir.LLVMLinkageInternal)
	case "linkonce":
		linkageAttr = mlir.NewLLVMLinkageAttr(b.ctx, mlir.LLVMLinkageLinkonce)
	case "linkonce_odr":
		linkageAttr = mlir.NewLLVMLinkageAttr(b.ctx, mlir.LLVMLinkageLinkonceODR)
	case "weak":
		linkageAttr = mlir.NewLLVMLinkageAttr(b.ctx, mlir.LLVMLinkageWeak)
	case "weak_odr":
		linkageAttr = mlir.NewLLVMLinkageAttr(b.ctx, mlir.LLVMLinkageWeakODR)
	default:
		linkageAttr = mlir.NewLLVMLinkageAttr(b.ctx, mlir.LLVMLinkageExternal)
	}

	state.AddAttributes(
		b.namedOf("function_type", mlir.NewTypeAttr(data.mlirType)),
		b.namedOf("sym_name", mlir.NewStringAttr(b.config.Ctx, data.linkname)),
		b.namedOf("sym_visibility", mlir.NewStringAttr(b.config.Ctx, visibility)),
		b.namedOf("llvm.linkage", linkageAttr),
	)

	for _, attr := range data.attributes {
		state.AddAttributes(b.namedOf(attr, mlir.NewUnitAttr(b.ctx)))
	}

	if data.isPackageInit {
		state.AddAttributes(
			b.namedOf("package_initializer", mlir.NewUnitAttr(b.ctx)),
			b.namedOf("priority", b.int32Attr(int32(data.priority))),
		)
	}

	// Create the operation, but don't add it to the module yet.
	funcOp := state.Create()

	// This operation will be added later safely.
	if isForwardDeclaration {
		b.forwardDeclarationsMutex.Lock()
		b.forwardDeclarations[data.linkname] = funcOp
		b.forwardDeclarationsMutex.Unlock()
	} else {
		b.addToModuleMutex.Lock()
		b.addToModule[data.linkname] = funcOp
		b.addToModuleMutex.Unlock()
	}
}

// findFuncInstance searches for an existing function instance matching the given
// typeMap. Returns nil if no match is found. Uses a read lock only.
func (b *Builder) findFuncInstance(data *funcData, typeMap TypeParamMap) *funcData {
	data.mutex.RLock()
	defer data.mutex.RUnlock()
	for _, instanceData := range data.instances {
		if len(typeMap) != len(instanceData.typeMap) {
			continue
		}
		match := true
		for key, T := range typeMap {
			otherT, ok := instanceData.typeMap[key]
			if !ok || !types.Identical(T, otherT) {
				match = false
				break
			}
		}
		if match {
			return instanceData
		}
	}
	return nil
}

func (b *Builder) createFuncInstance(ctx context.Context, signature *types.Signature, data *funcData, typeMap TypeParamMap) *funcData {
	data.mutex.Lock()
	defer data.mutex.Unlock()

	// Look up an existing instantiation.
	for _, instanceData := range data.instances {
		if len(typeMap) != len(instanceData.typeMap) {
			continue
		}
		match := true
		for key, T := range typeMap {
			otherT, ok := instanceData.typeMap[key]
			if !ok || !types.Identical(T, otherT) {
				match = false
				break
			}
		}
		if match {
			return instanceData
		}
	}

	targs := make([]types.Type, len(typeMap))
	for index, typ := range typeMap {
		targs[index] = typ
	}

	// Create the function data for this instance.
	instanceNo := len(data.instances)
	instanceData := &funcData{
		symbol:         fmt.Sprintf("%s$instance_%d", data.symbol, instanceNo),
		linkname:       fmt.Sprintf("%s$instance_%d", data.linkname, instanceNo),
		locals:         map[types.Object]Value{},
		scope:          data.scope,
		funcType:       data.funcType,
		signature:      signature,
		anonymousFuncs: data.anonymousFuncs,
		freeVars:       data.freeVars,
		recv:           data.recv,
		body:           data.body,
		pos:            data.pos,
		typeMap:        typeMap,
		isExported:     data.isExported,
		isInstance:     true,
		instance:       instanceNo,
		info:           data.info,
	}

	// Create the instantiated function type.
	instanceData.mlirType = b.GetType(newContextWithTypeMap(ctx, typeMap), signature).(goir.FunctionType)

	// Emit the instance.
	b.addToJobQueue(ctx, instanceData)

	// Cache this instantiation and return the data.
	data.instances = append(data.instances, instanceData)
	return instanceData
}

func (b *Builder) createFunctionValue(ctx context.Context, fn mlir.ValueLike, args mlir.ValueLike, location mlir.LocationLike) mlir.Value {
	// Examine the input function value.
	fnT := fn.Type()
	if ptrT, ok := goir.AsPointerType(fnT); ok {
		elementT := ptrT.ElementType()
		if !elementT.IsNull() && goir.TypeIsAFunctionType(elementT) {
			// Cast to an opaque pointer.
			fn = b.bitcastTo(ctx, fn, b.ptr, location)
		}
	} else {
		panic("function value is not a pointer")
	}

	// Create the function value.
	zeroOp := goir.NewZeroOperation(b.ctx, b._func, location)
	appendOperation(ctx, zeroOp)

	// Insert the function pointer.
	insertOp := goir.NewInsertOperation(b.ctx, 0, fn, resultOf(zeroOp), b._func, location)
	appendOperation(ctx, insertOp)

	if args != nil && !args.IsNull() {
		if ptrT, ok := goir.AsPointerType(args.Type()); ok && !ptrT.ElementType().IsNull() {
			args = b.bitcastTo(ctx, args, b.ptr, location)
		}

		// Insert the argument pack pointer value.
		insertOp = goir.NewInsertOperation(b.ctx, 1, args, resultOf(insertOp), b._func, location)
		appendOperation(ctx, insertOp)
	}

	// Return the struct value.
	return resultOf(insertOp).AsValue()
}

func (b *Builder) createThunk2(ctx context.Context, symbol string, callee string, signature *types.Signature, argTypes []types.Type, hasReceiver bool) {
	b.thunkMutex.Lock()
	defer b.thunkMutex.Unlock()

	// Look up the thunk in the symbol table first.
	if _, ok := b.thunks[symbol]; !ok {
		// Create the argument struct type.
		vars := make([]*types.Var, len(argTypes))
		for i := range argTypes {
			vars[i] = types.NewVar(token.NoPos, nil, fmt.Sprintf("arg$%d", i), argTypes[i])
		}

		argsT := types.NewStruct(vars, nil)
		argsPtrType := b.GetStoredType(ctx, types.NewPointer(argsT))

		nArgs := len(argTypes)
		if hasReceiver {
			// Exclude the receiver from the count
			nArgs--
		}

		// Any argument excluded from the argument pack MUST be passed to the resulting thunk directly.
		paramTypes := []mlir.TypeLike{argsPtrType}
		paramVars := []*types.Var{types.NewVar(token.NoPos, nil, fmt.Sprintf("param$%d", 0), argsT)}
		for i := nArgs; i < signature.Params().Len(); i++ {
			paramT := signature.Params().At(i).Type()
			paramTypes = append(paramTypes, b.GetStoredType(ctx, paramT))
			paramVars = append(paramVars, types.NewVar(token.NoPos, nil, fmt.Sprintf("param$%d", i+1), paramT))
		}
		paramLocs := make([]mlir.LocationLike, len(paramTypes))
		fill(paramLocs, b._noLoc)

		// Collect the result types.
		resultTypes := make([]mlir.TypeLike, 0, signature.Results().Len())
		resultVars := make([]*types.Var, 0, signature.Results().Len())
		for i := 0; i < signature.Results().Len(); i++ {
			resultT := signature.Results().At(i).Type()
			resultTypes = append(resultTypes, b.GetStoredType(ctx, resultT))
			resultVars = append(resultVars, types.NewVar(token.NoPos, nil, fmt.Sprintf("result$%d", i), resultT))
		}

		syntheticSig := types.NewSignatureType(nil, nil, nil,
			types.NewTuple(paramVars...), types.NewTuple(resultVars...), false)

		// Create thunk to wrap the method call.
		region := mlir.NewRegion()
		ctx = newContextWithRegion(ctx, region)

		entryBlock := mlir.NewBlock(paramTypes, paramLocs)
		region.AppendOwnedBlock(entryBlock)
		buildBlock(ctx, entryBlock, func() {
			argPackPtrValue := entryBlock.Argument(0)
			args := b.unpackArgPack(ctx, argTypes, argPackPtrValue, b._noLoc)

			// Gather the remaining arguments
			for i := 1; i < entryBlock.NumArguments(); i++ {
				args = append(args, entryBlock.Argument(i))
			}

			// Call the method.
			callOp := goir.NewCallOperation(b.ctx, callee, resultTypes, args, b._noLoc)
			appendOperation(ctx, callOp)

			// Return the results.
			returnOp := goir.NewReturnOperation(b.ctx, resultsOf(callOp), b._noLoc)
			appendOperation(ctx, returnOp)
		})

		// Create the function operation for this thunk.
		thunkFuncType := b.GetType(ctx, syntheticSig)
		funcOp := mlir.NewOperationState("go.func", b._noLoc).
			AddOwnedRegions(region).
			AddAttributes(
				b.namedOf("function_type", mlir.NewTypeAttr(thunkFuncType)),
				b.namedOf("sym_name", mlir.NewStringAttr(b.ctx, symbol)),
				b.namedOf("sym_visibility", mlir.NewStringAttr(b.ctx, "private")),
			).Create()

		// This operation will be added later safely.
		b.addToModuleMutex.Lock()
		b.addToModule[symbol] = funcOp
		b.addToModuleMutex.Unlock()
		b.thunks[symbol] = struct{}{}
	}
}

func (b *Builder) createArgumentPack(ctx context.Context, values []mlir.ValueLike, valueTypes []types.Type, location mlir.LocationLike) (mlir.ValueLike, mlir.TypeLike, mlir.TypeLike) {
	if len(values) == 0 {
		return nil, nil, nil
	}

	if len(values) != len(valueTypes) {
		panic("number of value expressions and types must match")
	}

	vars := make([]*types.Var, len(values))
	for i := range len(values) {
		vars[i] = types.NewVar(token.NoPos, nil, fmt.Sprintf("arg$%d", i), valueTypes[i])
	}

	valuesT := types.NewStruct(vars, nil)
	valuesPtrT := b.GetStoredType(ctx, types.NewPointer(valuesT))
	valuesType := b.GetStoredType(ctx, valuesT)

	// Create the values struct.
	zeroOp := goir.NewZeroOperation(b.ctx, valuesType, location)
	appendOperation(ctx, zeroOp)
	argsValue := resultOf(zeroOp)
	for i, value := range values {
		insertOp := goir.NewInsertOperation(b.ctx, uint64(i), value, argsValue, valuesType, location)
		appendOperation(ctx, insertOp)
		argsValue = resultOf(insertOp)
	}

	return argsValue, valuesType, valuesPtrT
}

func (b *Builder) unpackArgPack(ctx context.Context, argTypes []types.Type, pack mlir.ValueLike, location mlir.LocationLike) []mlir.ValueLike {
	result := make([]mlir.ValueLike, len(argTypes))
	ptrT, _ := goir.AsPointerType(pack.Type())
	argPackT := ptrT.ElementType()
	for i, T := range argTypes {
		argT := b.GetStoredType(ctx, T)
		argPtrT := b.GetStoredType(ctx, types.NewPointer(T))
		gepOp := goir.NewGepOperation(b.ctx, pack, argPackT, []int{0, i}, nil, []bool{false, false}, argPtrT, location)
		appendOperation(ctx, gepOp)
		result[i] = b.emitLoad(ctx, resultOf(gepOp), argT, location)
	}
	return result
}

func (b *Builder) emitBuiltinCallWrapper(ctx context.Context, ident *ast.Ident) string {
	b.builtinWrapperMutex.Lock()
	defer b.builtinWrapperMutex.Unlock()

	if symbol, ok := b.builtinWrappers[ident.Name]; ok {
		return symbol
	}

	// Emit a wrapper function for this builtin.
	signature := b.typeOf(ctx, ident).(*types.Signature)

	// Collect the argument types.
	paramTypes := make([]mlir.TypeLike, signature.Params().Len())
	for i := 0; i < signature.Params().Len(); i++ {
		paramTypes[i] = b.GetStoredType(ctx, signature.Params().At(i).Type())
	}

	paramLocs := make([]mlir.LocationLike, len(paramTypes))
	fill(paramLocs, b._noLoc)

	// Collect the result types.
	resultTypes := make([]mlir.TypeLike, signature.Results().Len())
	for i := 0; i < signature.Results().Len(); i++ {
		resultTypes[i] = b.GetStoredType(ctx, signature.Results().At(i).Type())
	}

	// Create the wrapper function body.
	region := mlir.NewRegion()
	ctx = newContextWithRegion(ctx, region)

	entryBlock := mlir.NewBlock(paramTypes, paramLocs)
	region.AppendOwnedBlock(entryBlock)
	buildBlock(ctx, entryBlock, func() {
		args := make([]mlir.ValueLike, signature.Params().Len())
		for i := 0; i < signature.Params().Len(); i++ {
			args[i] = entryBlock.Argument(i)
		}

		// Emit the builtin call into the wrapper function.
		op := goir.NewBuiltInCallOperation(b.ctx, ident.Name, resultTypes, args, b._noLoc)
		appendOperation(ctx, op)

		// Return the results.
		returnOp := goir.NewReturnOperation(b.ctx, resultsOf(op), b._noLoc)
		appendOperation(ctx, returnOp)
	})

	// Create the function operation.
	symbol := fmt.Sprintf("_builtin_wrapper_%s", ident.Name)
	wrapperFuncT := b.GetType(ctx, signature)
	funcOp := mlir.NewOperationState("go.func", b._noLoc).
		AddOwnedRegions(region).AddAttributes(
		b.namedOf("function_type", mlir.NewTypeAttr(wrapperFuncT)),
		b.namedOf("sym_name", mlir.NewStringAttr(b.ctx, symbol)),
		b.namedOf("sym_visibility", mlir.NewStringAttr(b.ctx, "private")),
	).Create()

	// This operation will be added later safely.
	b.addToModuleMutex.Lock()
	b.addToModule[symbol] = funcOp
	b.addToModuleMutex.Unlock()
	b.builtinWrappers[ident.Name] = symbol
	return symbol
}
