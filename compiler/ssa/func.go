package ssa

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"sync"

	"pkg.si-go.dev/sigo/mlir"
)

type funcData struct {
	symbol      string
	linkname    string
	scope       *types.Scope
	funcType    *ast.FuncType
	mlirType    mlir.Type
	signature   *types.Signature
	freeVars    []*FreeVar
	contextType mlir.Type
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
	t mlir.Type
	l mlir.Location
}

type inputParams []inputParam

func (i inputParams) types() []mlir.Type {
	result := make([]mlir.Type, 0, len(i))
	for _, i := range i {
		result = append(result, i.t)
	}
	return result
}

func (i inputParams) locations() []mlir.Location {
	result := make([]mlir.Location, 0, len(i))
	for _, i := range i {
		result = append(result, i.l)
	}
	return result
}

func (f *funcData) createContextStructValue(ctx context.Context, b *Builder, location mlir.Location) (mlir.Value, mlir.Type) {
	// Collect the addresses of each value captured by this function.
	var values []mlir.Value
	for _, fv := range f.freeVars {
		ptr := b.lookupValue(ctx, fv.obj).Pointer(ctx, location)
		values = append(values, ptr)
	}
	return b.createArgumentPack(ctx, values, location)
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
				loc = mlir.LocationFusedGet(b.ctx, []mlir.Location{loc}, compileUnitAttr)
			}
		}
	}

	// Create the function operation.
	state := mlir.OperationStateGet("go.func", loc)

	argOffset := 0

	// Determine this number of inputs required to call this function.
	numInputs := mlir.GoFunctionTypeGetNumInputs(data.mlirType)
	if data.signature.Recv() != nil {
		numInputs++
	}
	inputs := make(inputParams, numInputs)

	// Collect input information.
	if data.signature.Recv() != nil {
		// The receiver is the first parameter to this function. So, offset by 1.
		argOffset = 1
		inputs[0].t = mlir.GoFunctionTypeGetReceiver(data.mlirType)
		inputs[0].l = b.location(ctx, data.signature.Recv().Pos())
	}

	for i := 0; i < data.signature.Params().Len(); i++ {
		param := data.signature.Params().At(i)
		inputs[argOffset+i].t = mlir.GoFunctionTypeGetInput(data.mlirType, i)
		inputs[argOffset+i].l = b.location(ctx, param.Pos())
	}

	if data.isAnonymous {
		// Don't emit a local variable for the context pointer.
		argOffset = 1
	}

	// Create the region in which all blocks will be placed in.
	region := mlir.RegionCreate()
	ctx = newContextWithRegion(ctx, region)
	mlir.OperationStateAddOwnedRegions(state, []mlir.Region{region})

	// NOTE: Forward declarations will not have any block.
	if !isForwardDeclaration {
		// Create the entry block for the current function.
		entryBlock := mlir.BlockCreate2(inputs.types(), inputs.locations())
		mlir.RegionAppendOwnedBlock(region, entryBlock)

		ctx = newContextWithCurrentBlock(ctx)
		setCurrentBlock(ctx, entryBlock)

		// NOTE: The captures are part of the types.Signature object for this function.
		data.mutex.RLock()
		if len(data.freeVars) > 0 {
			// Update free variable pointers.
			ctxValue := mlir.BlockGetArgument(entryBlock, 0)
			for i, fv := range data.freeVars {
				// Append the freevar's alloca operation to the current block.
				allocaOp := mlir.ValueGetDefiningOperation(fv.ptr)
				mlir.GoAllocaOperationSetName(allocaOp, fv.obj.Name())
				appendOperation(ctx, allocaOp)

				ptrType := mlir.GoCreatePointerType(fv.T)

				// GEP into the context to derive the address of the free variable.
				gepOp := mlir.GoCreateGepOperation2(b.ctx, ctxValue, data.contextType, []any{0, i}, mlir.GoCreatePointerType(ptrType), loc)
				appendOperation(ctx, gepOp)

				// Load the address of the external local variable.
				loadOp := mlir.GoCreateLoadOperation(b.ctx, resultOf(gepOp), ptrType, loc)
				appendOperation(ctx, loadOp)

				// Store the address of the free variable at the address of the stack allocation.
				storeOp := mlir.GoCreateStoreOperation(b.ctx, resultOf(loadOp), fv.ptr, loc)
				appendOperation(ctx, storeOp)
			}

			// Function arguments start after the capture list.
			argOffset = 1
		}
		data.mutex.RUnlock()

		if data.recv != nil {
			for _, field := range data.recv.List {
				for _, name := range field.Names {
					recvVar := b.objectOf(ctx, name)
					recvVal := mlir.BlockGetArgument(entryBlock, 0)

					// Emit a local variable allocation to hold the argument value.
					addr := b.emitLocalVar(ctx, recvVar, mlir.ValueGetType(recvVal), true)

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
					argVal := mlir.BlockGetArgument(entryBlock, arg)
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
				labeledBlock := mlir.BlockCreate2(nil, nil)
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
			zeroValues := make([]mlir.Value, 0, data.signature.Results().Len())
			for i := 0; i < data.signature.Results().Len(); i++ {
				result := data.signature.Results().At(i)
				value := b.emitZeroValue(ctx, result.Type(), endLocation)
				zeroValues = append(zeroValues, value)
			}

			returnOp := mlir.GoCreateReturnOperation(b.ctx, zeroValues, endLocation)
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

	mlir.OperationStateAddAttributes(state, []mlir.NamedAttribute{
		b.namedOf("function_type", mlir.TypeAttrGet(data.mlirType)),
		b.namedOf("sym_name", mlir.StringAttrGet(b.config.Ctx, data.linkname)),
		b.namedOf("sym_visibility", mlir.StringAttrGet(b.config.Ctx, visibility)),
		b.namedOf("llvm.linkage", mlir.GetLLVMLinkageAttr(b.ctx, linkage)),
		b.namedOf("passthrough", b.strArrayAttr(data.attributes...)),
	})

	if data.isPackageInit {
		mlir.OperationStateAddAttributes(state, []mlir.NamedAttribute{
			b.namedOf("package_initializer", mlir.UnitAttrGet(b.ctx)),
			b.namedOf("priority", b.int32Attr(int32(data.priority))),
		})
	}

	// Create the operation, but don't add it to the module yet.
	funcOp := mlir.OperationCreate(state)

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

	// Create a new instantiated signature.
	var recv *types.Var
	var params []*types.Var
	var results []*types.Var

	var convertType func(types.Type) types.Type
	convertType = func(T types.Type) types.Type {
		switch T := T.(type) {
		case *types.Array:
			return types.NewArray(convertType(T.Elem()), T.Len())
		case *types.Chan:
			return types.NewChan(T.Dir(), convertType(T.Elem()))
		case *types.Map:
			return types.NewMap(convertType(T.Key()), convertType(T.Elem()))
		case *types.Pointer:
			return types.NewPointer(convertType(T.Elem()))
		case *types.Slice:
			return types.NewSlice(convertType(T.Elem()))
		case *types.TypeParam:
			return typeMap[T.Index()]
		default:
			return T
		}
	}

	if signature.Recv() != nil {
		src := signature.Recv()
		T := convertType(src.Type())
		recv = types.NewParam(src.Pos(), src.Pkg(), src.Name(), T)
	}

	for src := range signature.Params().Variables() {
		T := convertType(src.Type())
		param := types.NewParam(src.Pos(), src.Pkg(), src.Name(), T)
		params = append(params, param)
	}

	for src := range signature.Results().Variables() {
		T := convertType(src.Type())
		result := types.NewVar(src.Pos(), src.Pkg(), src.Name(), T)
		results = append(results, result)
	}

	newSignature := types.NewSignatureType(recv, nil, nil, types.NewTuple(params...), types.NewTuple(results...),
		signature.Variadic())

	// Create the function data for this instance.
	instanceNo := len(data.instances)
	instanceData := &funcData{
		symbol:         fmt.Sprintf("%s$instance_%d", data.symbol, instanceNo),
		linkname:       fmt.Sprintf("%s$instance_%d", data.linkname, instanceNo),
		locals:         map[types.Object]Value{},
		scope:          data.scope,
		funcType:       data.funcType,
		signature:      newSignature,
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
	instanceData.mlirType = b.createSignatureType(newContextWithTypeMap(ctx, typeMap), newSignature)

	// Emit the instance.
	b.addToJobQueue(ctx, instanceData)

	// Cache this instantiation and return the data.
	data.instances = append(data.instances, instanceData)
	return instanceData
}

func (b *Builder) createFunctionValue(ctx context.Context, fn mlir.Value, args mlir.Value, location mlir.Location) mlir.Value {
	// Examine the input function value.
	fnT := mlir.ValueGetType(fn)
	if mlir.GoTypeIsAPointer(fnT) {
		elementT := mlir.GoPointerTypeGetElementType(fnT)
		if !mlir.TypeIsNull(elementT) && mlir.GoTypeIsAFunctionType(elementT) {
			// Cast to an opaque pointer.
			fn = b.bitcastTo(ctx, fn, b.ptr, location)
		}
	} else {
		panic("function value is not a pointer")
	}

	// Create the function value.
	zeroOp := mlir.GoCreateZeroOperation(b.ctx, b._func, location)
	appendOperation(ctx, zeroOp)

	// Insert the function pointer.
	insertOp := mlir.GoCreateInsertOperation(b.ctx, 0, fn, resultOf(zeroOp), b._func, location)
	appendOperation(ctx, insertOp)

	if args != nil {
		if !mlir.TypeIsNull(mlir.GoPointerTypeGetElementType(mlir.ValueGetType(args))) {
			args = b.bitcastTo(ctx, args, b.ptr, location)
		}

		// Insert the argument pack pointer value.
		insertOp = mlir.GoCreateInsertOperation(b.ctx, 1, args, resultOf(insertOp), b._func, location)
		appendOperation(ctx, insertOp)
	}

	// Return the struct value.
	return resultOf(insertOp)
}

func (b *Builder) createThunk(ctx context.Context, symbol string, callee string, signature *types.Signature, argTypes []mlir.Type, hasReceiver bool) {
	b.thunkMutex.Lock()
	defer b.thunkMutex.Unlock()

	// Look up the thunk in the symbol table first.
	if _, ok := b.thunks[symbol]; !ok {
		// Create the argument struct type.
		argsType := mlir.GoCreateBasicStructType(b.ctx, argTypes)

		nArgs := len(argTypes)
		if hasReceiver {
			// Exclude the receiver from the count
			nArgs--
		}

		// Any argument excluded from the argument pack MUST be passed to the resulting thunk directly.
		paramTypes := []mlir.Type{mlir.GoCreatePointerType(argsType)}
		for i := nArgs; i < signature.Params().Len(); i++ {
			paramTypes = append(paramTypes, b.GetStoredType(ctx, signature.Params().At(i).Type()))
		}
		paramLocs := make([]mlir.Location, len(paramTypes))
		fill(paramLocs, b._noLoc)

		// Collect the result types.
		resultTypes := make([]mlir.Type, 0, signature.Results().Len())
		for i := 0; i < signature.Results().Len(); i++ {
			resultTypes = append(resultTypes, b.GetStoredType(ctx, signature.Results().At(i).Type()))
		}

		// Create thunk to wrap the method call.
		region := mlir.RegionCreate()
		ctx = newContextWithRegion(ctx, region)

		entryBlock := mlir.BlockCreate2(paramTypes, paramLocs)
		mlir.RegionAppendOwnedBlock(region, entryBlock)
		buildBlock(ctx, entryBlock, func() {
			argPackPtrValue := mlir.BlockGetArgument(entryBlock, 0)
			args := b.unpackArgPack(ctx, argTypes, argPackPtrValue, b._noLoc)

			// Gather the remaining arguments
			for i := 1; i < mlir.BlockGetNumArguments(entryBlock); i++ {
				args = append(args, mlir.BlockGetArgument(entryBlock, i))
			}

			// Call the method.
			callOp := mlir.GoCreateCallOperation(b.ctx, callee, resultTypes, args, b._noLoc)
			appendOperation(ctx, callOp)

			// Return the results.
			returnOp := mlir.GoCreateReturnOperation(b.ctx, resultsOf(callOp), b._noLoc)
			appendOperation(ctx, returnOp)
		})

		// Create the function operation for this thunk.
		thunkFuncType := mlir.GoCreateFunctionType(b.ctx, nil, paramTypes, resultTypes)
		state := mlir.OperationStateGet("go.func", b._noLoc)
		mlir.OperationStateAddOwnedRegions(state, []mlir.Region{region})
		mlir.OperationStateAddAttributes(state, []mlir.NamedAttribute{
			b.namedOf("function_type", mlir.TypeAttrGet(thunkFuncType)),
			b.namedOf("sym_name", mlir.StringAttrGet(b.ctx, symbol)),
			b.namedOf("sym_visibility", mlir.StringAttrGet(b.ctx, "private")),
		})

		funcOp := mlir.OperationCreate(state)

		// This operation will be added later safely.
		b.addToModuleMutex.Lock()
		b.addToModule[symbol] = funcOp
		b.addToModuleMutex.Unlock()
		b.thunks[symbol] = struct{}{}
	}
}

func (b *Builder) createArgumentPack(ctx context.Context, args []mlir.Value, location mlir.Location) (mlir.Value, mlir.Type) {
	if len(args) == 0 {
		return nil, nil
	}

	// Collect the argument types.
	argTypes := make([]mlir.Type, len(args))
	for i := range args {
		argTypes[i] = mlir.ValueGetType(args[i])
	}

	// Create the argument struct.
	argsType := mlir.GoCreateBasicStructType(b.ctx, argTypes)
	zeroOp := mlir.GoCreateZeroOperation(b.ctx, argsType, location)
	appendOperation(ctx, zeroOp)
	argsValue := resultOf(zeroOp)
	for i, arg := range args {
		insertOp := mlir.GoCreateInsertOperation(b.ctx, uint64(i), arg, argsValue, argsType, location)
		appendOperation(ctx, insertOp)
		argsValue = resultOf(insertOp)
	}
	return argsValue, argsType
}

func (b *Builder) unpackArgPack(ctx context.Context, argTypes []mlir.Type, pack mlir.Value, location mlir.Location) []mlir.Value {
	result := make([]mlir.Value, len(argTypes))
	argPackT := mlir.GoPointerTypeGetElementType(mlir.ValueGetType(pack))
	for i, T := range argTypes {
		gepOp := mlir.GoCreateGepOperation2(b.ctx, pack, argPackT, []any{0, i}, mlir.GoCreatePointerType(T), location)
		appendOperation(ctx, gepOp)
		loadOp := mlir.GoCreateLoadOperation(b.ctx, resultOf(gepOp), T, location)
		appendOperation(ctx, loadOp)
		result[i] = resultOf(loadOp)
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
	paramTypes := make([]mlir.Type, signature.Params().Len())
	for i := 0; i < signature.Params().Len(); i++ {
		paramTypes[i] = b.GetStoredType(ctx, signature.Params().At(i).Type())
	}

	paramLocs := make([]mlir.Location, len(paramTypes))
	fill(paramLocs, b._noLoc)

	// Collect the result types.
	resultTypes := make([]mlir.Type, signature.Results().Len())
	for i := 0; i < signature.Results().Len(); i++ {
		resultTypes[i] = b.GetStoredType(ctx, signature.Results().At(i).Type())
	}

	// Create the wrapper function body.
	region := mlir.RegionCreate()
	ctx = newContextWithRegion(ctx, region)

	entryBlock := mlir.BlockCreate2(paramTypes, paramLocs)
	mlir.RegionAppendOwnedBlock(region, entryBlock)
	buildBlock(ctx, entryBlock, func() {
		args := make([]mlir.Value, signature.Params().Len())
		for i := 0; i < signature.Params().Len(); i++ {
			args[i] = mlir.BlockGetArgument(entryBlock, i)
		}

		// Emit the builtin call into the wrapper function.
		op := mlir.GoCreateBuiltInCallOperation(b.ctx, ident.Name, resultTypes, args, b._noLoc)
		appendOperation(ctx, op)

		// Return the results.
		returnOp := mlir.GoCreateReturnOperation(b.ctx, resultsOf(op), b._noLoc)
		appendOperation(ctx, returnOp)
	})

	// Create the function operation.
	symbol := fmt.Sprintf("_builtin_wrapper_%s", ident.Name)
	wrapperFuncT := b.createSignatureType(ctx, signature)
	state := mlir.OperationStateGet("go.func", b._noLoc)
	mlir.OperationStateAddOwnedRegions(state, []mlir.Region{region})
	mlir.OperationStateAddAttributes(state, []mlir.NamedAttribute{
		b.namedOf("function_type", mlir.TypeAttrGet(wrapperFuncT)),
		b.namedOf("sym_name", mlir.StringAttrGet(b.ctx, symbol)),
		b.namedOf("sym_visibility", mlir.StringAttrGet(b.ctx, "private")),
	})

	funcOp := mlir.OperationCreate(state)

	// This operation will be added later safely.
	b.addToModuleMutex.Lock()
	b.addToModule[symbol] = funcOp
	b.addToModuleMutex.Unlock()
	b.builtinWrappers[ident.Name] = symbol
	return symbol
}
