package ssa

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"sync"

	"omibyte.io/sigo/mlir"
)

type funcData struct {
	symbol      string
	scope       *types.Scope
	funcType    *ast.FuncType
	mlirType    mlir.Type
	signature   *types.Signature
	freeVars    []*FreeVar
	recv        *ast.FieldList
	body        *ast.BlockStmt
	pos         token.Pos
	isGeneric   bool
	isExported  bool
	isInstance  bool
	isAnonymous bool

	isPackageInit bool
	priority      int

	mutex    sync.RWMutex
	instance int

	//locals         map[string]Value
	locals         map[types.Object]Value
	anonymousFuncs map[*ast.FuncLit]*funcData
	instances      map[*types.Signature]*funcData
	typeMap        map[int]types.Type

	decl *ast.FuncDecl
	info *types.Info
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
		//ptr := b.valueOf(ctx, fv.ident).Pointer(ctx, location)
		ptr := b.lookupValue(fv.obj).Pointer(ctx, location)
		values = append(values, ptr)
	}
	return b.createArgumentPack(ctx, values, location)
}

func (b *Builder) emitFunc(ctx context.Context, data *funcData) {
	if data.isGeneric {
		// Do not attempt to generate uninstantiated generic functions.
		return
	}

	var queue *jobQueue
	if val := ctx.Value(jobQueueKey{}); val != nil {
		queue = val.(*jobQueue)
	}

	// Set the current data in a fresh context.
	ctx = newContextWithFuncData(context.Background(), data)
	ctx = newContextWithInfo(ctx, data.info)

	if queue != nil {
		ctx = context.WithValue(ctx, jobQueueKey{}, queue)
	}

	// Get the location of the input function.
	loc := b.location(data.pos)

	// Fuse the location with the compile unit if applicable.
	if data.pos.IsValid() {
		if file := b.config.Fset.File(data.pos); file != nil {
			if compileUnitAttr, ok := b.compileUnits[file]; ok {
				loc = mlir.LocationFusedGet(b.ctx, []mlir.Location{loc}, compileUnitAttr)
			}
		}
	}

	// Create the region in which all blocks will be placed in.
	region := mlir.RegionCreate()
	ctx = newContextWithRegion(ctx, region)

	// Collect input information.
	argOffset := 0
	inputs := make(inputParams, mlir.FunctionTypeGetNumInputs(data.mlirType))
	for i := 0; i < len(inputs); i++ {
		inputs[i].t = mlir.FunctionTypeGetInput(data.mlirType, i)
	}

	if data.signature.Recv() != nil {
		// The receiver is the first parameter to this function. So, offset by 1.
		argOffset = 1
		inputs[0].l = b.location(data.signature.Recv().Pos())
	}

	for i := 0; i < data.signature.Params().Len(); i++ {
		param := data.signature.Params().At(i)
		inputs[argOffset+i].l = b.location(param.Pos())
	}

	if data.isAnonymous {
		// Don't emit a local variable for the context pointer.
		argOffset = 1
	}

	// NOTE: Forward declarations will not have any block.
	if data.body != nil {
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
				gepOp := mlir.GoCreateGepOperation2(b.ctx, ctxValue, fv.T, []any{0, i}, mlir.GoCreatePointerType(ptrType), loc)
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
					addr := b.emitLocalVar(ctx, recvVar, mlir.ValueGetType(recvVal))

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
					addr := b.emitLocalVar(ctx, argVar, b.GetStoredType(ctx, argVar.Type()))

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
					b.emitLocalVar(ctx, resultVar, b.GetStoredType(ctx, resultVar.Type()))
				}
			}
		}

		// Create all labeled blocks before emitting the body since some statements might need to be able to look them
		// up later. The ast.LabeledStmt will actually append them to the current region as the appropriate time.
		labeledBlocks := map[string]mlir.Block{}
		for _, stmt := range data.body.List {
			if stmt, ok := stmt.(*ast.LabeledStmt); ok {
				labeledBlock := mlir.BlockCreate2(nil, nil)
				labeledBlocks[stmt.Label.Name] = labeledBlock
			}
		}
		ctx = newContextWithLabeledBlocks(ctx, labeledBlocks)

		// Fill the function body.
		b.emitBlock(ctx, data.body)

		// Assume that the current block is the "last" logical block in the function.
		lastBlock := currentBlock(ctx)

		if !blockHasTerminator(lastBlock) {
			// Control flow fell off the end of the function block. Insert tail operations.
			//TODO: Run defers.

			endLocation := b.location(data.body.End())
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

	// Create the function operation.
	state := mlir.OperationStateGet("func.func", loc)
	mlir.OperationStateAddOwnedRegions(state, []mlir.Region{region})
	visibility := "public"
	if !data.isExported || data.body == nil {
		// NOTE: Forward declarations MUST be private.
		visibility = "private"
	}

	mlir.OperationStateAddAttributes(state, []mlir.NamedAttribute{
		b.namedOf("function_type", mlir.TypeAttrGet(data.mlirType)),
		b.namedOf("sym_name", mlir.StringAttrGet(b.config.Ctx, data.symbol)),
		b.namedOf("sym_visibility", mlir.StringAttrGet(b.config.Ctx, visibility)),
	})

	if data.isPackageInit {
		mlir.OperationStateAddAttributes(state, []mlir.NamedAttribute{
			b.namedOf("package_initializer", mlir.UnitAttrGet(b.ctx)),
			b.namedOf("priority", mlir.IntegerAttrGet(mlir.IntegerTypeGet(b.ctx, 32), int64(data.priority))),
		})
	}

	// Create the operation, but don't add it to the module yet.
	funcOp := mlir.OperationCreate(state)

	// This operation will be added later safely.
	b.addToModuleMutex.Lock()
	b.addToModule[data.symbol] = funcOp
	b.addToModuleMutex.Unlock()
}

func (b *Builder) createFuncInstance(ctx context.Context, genericSignature *types.Signature, instance types.Instance, data *funcData) *funcData {
	var signature *types.Signature

	data.mutex.Lock()
	defer data.mutex.Unlock()

	// Look up an existing instantiation.
	if instanceData, ok := data.instances[signature]; ok {
		return instanceData
	}

	// Create the type map.
	typeMap := map[int]types.Type{}

	// Map the receiver type parameter to the concrete receiver type.
	if recv := genericSignature.Recv(); recv != nil {
		signature = genericSignature
		targs := signature.Recv().Type().(*types.Named).TypeArgs()
		tparams := signature.Recv().Type().(*types.Named).TypeParams()
		for i := 0; i < targs.Len(); i++ {
			typeMap[tparams.At(i).Index()] = targs.At(i)
		}
	} else {
		signature = instance.Type.(*types.Signature)

		// Map the type parameters to the concrete types.
		for i := 0; i < genericSignature.TypeParams().Len(); i++ {
			param := genericSignature.TypeParams().At(i)
			typeMap[param.Index()] = instance.TypeArgs.At(i)
		}
	}

	// Create the function data for this instance.
	instanceNo := len(data.instances)
	instanceData := &funcData{
		symbol:         fmt.Sprintf("%s$instance_%d", data.symbol, instanceNo),
		locals:         data.locals,
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
	ctx = newContextWithFuncData(ctx, instanceData)
	instanceData.mlirType = b.createSignatureType(ctx, signature, false)

	// Emit the instance.
	b.emitFunc(ctx, instanceData)

	// Cache this instantiation and return the data.
	data.instances[signature] = instanceData
	return instanceData
}

func (b *Builder) createFunctionValue(ctx context.Context, fn mlir.Value, args mlir.Value, location mlir.Location) mlir.Value {
	// Create the function value.
	zeroOp := mlir.GoCreateZeroOperation(b.ctx, b._func, location)
	appendOperation(ctx, zeroOp)

	if mlir.TypeIsAFunction(mlir.ValueGetType(fn)) {
		// Bitcast the function value to a pointer.
		fn = b.bitcastTo(ctx, fn, b.ptr, location)
	}

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
		argsType := mlir.GoCreateLiteralStructType(b.ctx, argTypes)

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
		thunkFuncType := mlir.FunctionTypeGet(b.ctx, paramTypes, resultTypes)
		state := mlir.OperationStateGet("func.func", b._noLoc)
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
	// Collect the argument types.
	argTypes := make([]mlir.Type, len(args))
	for i := range args {
		argTypes[i] = mlir.ValueGetType(args[i])
	}

	// Create the argument struct.
	argsType := mlir.GoCreateLiteralStructType(b.ctx, argTypes)
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
