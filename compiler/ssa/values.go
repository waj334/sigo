package ssa

import (
	"context"
	"go/ast"
	"go/types"

	"pkg.si-go.dev/go-mlir/mlir"
	"pkg.si-go.dev/sigo/compiler/ssa/internal/asset"
	"pkg.si-go.dev/sigo/goir/binding/goir"
)

func (b *Builder) valueOf(ctx context.Context, node ast.Node) Value {
	if node == nil {
		return nil
	}

	switch node := node.(type) {
	case *ast.SelectorExpr:
		return b.NewTempValue(b.emitSelectAddr(ctx, node))
	case *ast.IndexExpr:
		return b.NewTempValue(b.emitIndexAddr(ctx, node))
	case *ast.Ident:
		obj := b.objectOf(ctx, node)

		// Look in the current function's locals first.
		if data := currentFuncData(ctx); data != nil {
			data.mutex.RLock()
			if obj == nil {
				data.mutex.RUnlock()
				return nil
			}

			if value, ok := data.locals[obj]; ok {
				data.mutex.RUnlock()
				return value
			}
			data.mutex.RUnlock()
		}

		// Look up the value by object.
		return b.lookupValue(ctx, obj)
	default:
		return nil
	}
}

func (b *Builder) lookupValue(ctx context.Context, obj types.Object) Value {
	// Lock the value cache mutex for reading.
	b.valueCacheMutex.RLock()
	defer b.valueCacheMutex.RUnlock()
	value, ok := b.valueCache[obj]
	if ok {
		return value
	}

	data := currentFuncData(ctx)
	if data != nil {
		data.mutex.RLock()
		defer data.mutex.RUnlock()
		return data.locals[obj]
	}

	return nil
}

func (b *Builder) emitLocalVar(ctx context.Context, obj types.Object, T mlir.TypeLike, isArg bool) *LocalValue {
	// Allocate memory for this local variable on the stack.
	// NOTE: It may be determined later that this variable escapes to the heap and the following operation will be
	//       replaced by a heap allocation.
	location := b.location(ctx, obj.Pos())

	elementT := resolveType(ctx, obj.Type())
	ptrValue, _ := b.emitNamedAlloca(ctx, obj.Name(), T, elementT, location).AsResult()
	op := ptrValue.OwningOperation()
	if isArg {
		op.SetAttributeByName("isArgument", mlir.NewUnitAttr(b.ctx))
	}

	value := &LocalValue{
		ptr: ptrValue,
		T:   T,
		b:   b,
		obj: obj,
	}

	data := currentFuncData(ctx)
	if data == nil {
		b.valueCacheMutex.Lock()
		defer b.valueCacheMutex.Unlock()
		b.valueCache[obj] = value
	} else {
		data.mutex.Lock()
		data.locals[obj] = value
		data.mutex.Unlock()
	}

	return value
}

func (b *Builder) emitGlobalVar(ctx context.Context, ident *ast.Ident) *GlobalValue {
	obj := b.objectOf(ctx, ident).(*types.Var)
	symbol := qualifiedName(obj.Name(), obj.Pkg())
	info := b.config.Program.Symbols.GetSymbolInfo(symbol)
	T := b.GetStoredType(ctx, obj.Type())

	// Resolve the actual symbol name for this global.
	symbol = b.resolveSymbol(symbol)

	// Determine the linkage of the global variable.
	var linkage mlir.LLVMLinkageAttr
	if obj.Exported() || info.Exported || len(info.LinkName) > 0 {
		linkage = mlir.NewLLVMLinkageAttr(b.ctx, mlir.LLVMLinkageExternal)
	} else {
		linkage = mlir.NewLLVMLinkageAttr(b.ctx, mlir.LLVMLinkagePrivate)
	}

	// Fuse the location with the compile unit if applicable.
	location := b.location(ctx, obj.Pos())
	if file := b.config.Fset.File(obj.Pos()); file != nil {
		if compileUnitAttr, ok := b.compileUnits[file]; ok {
			location = mlir.NewFusedLoc(b.ctx, []mlir.LocationLike{location}, compileUnitAttr)
		}
	}

	section := info.Section
	_, isEmbdedded := b.config.Program.EmbedContents[symbol]
	if isEmbdedded {
		section = ""
	}

	// Emit the global variable.
	globalOp := goir.NewGlobalOperation(b.ctx, linkage, symbol, section, info.Alignment, T, location)
	b.appendToModule(globalOp)
	value := &GlobalValue{
		symbol: symbol,
		T:      T,
		GoT:    obj.Type(),
		ctx:    b.ctx,
		b:      b,
	}

	if info.Exported {
		// Mark this var as explicitly exported.
		globalOp.SetAttributeByName("go.exported", mlir.NewUnitAttr(b.ctx))
	}

	b.valueCacheMutex.Lock()
	defer b.valueCacheMutex.Unlock()

	b.valueCache[obj] = value
	b.addSymbol(globalOp)

	return value
}

func (b *Builder) emitCastPointerToInt(ctx context.Context, X mlir.ValueLike, location mlir.LocationLike) mlir.Value {
	op := goir.NewPtrToIntOperation(b.ctx, X, b.GetStoredType(ctx, types.Typ[types.Uintptr]), location)
	appendOperation(ctx, op)
	return resultOf(op).AsValue()
}

func (b *Builder) makeCopyOf(ctx context.Context, X mlir.ValueLike, XT types.Type, location mlir.LocationLike) mlir.Value {
	elementType := b.GetStoredType(ctx, XT)
	ptrType := b.GetStoredType(ctx, types.NewPointer(XT))

	// Taking the address in global initializers causes the allocation to escape to the heap.
	isHeap := false
	if isGlobalContext(ctx) {
		isHeap = true
	}

	// Allocate memory on the stack to hold the object.
	allocaOp := goir.NewAllocaOperation(b.ctx, ptrType, elementType, 1, isHeap, location)
	appendOperation(ctx, allocaOp)

	// Store the object at the address.
	b.emitStore(ctx, X, resultOf(allocaOp), location)

	// Return the address.
	return resultOf(allocaOp).AsValue()
}

func (b *Builder) emitNamedAlloca(ctx context.Context, name string, T mlir.TypeLike, GoT types.Type, location mlir.LocationLike) mlir.Value {
	// Create the pointer type through the type cache so that pointers to named
	// types use the same deferred pointer as the rest of the compiler.
	PT := b.pointerOf(ctx, GoT)

	// Allocate memory on the stack to hold the object.
	allocaOp := goir.NewAllocaOperation(b.ctx, PT, T, 1, false, location)
	appendOperation(ctx, allocaOp)

	// NOTE: Omitted identifiers ( `_` )  will not have any debug information attached.
	if len(name) > 0 && name != "_" {
		goir.AllocaOperationSetName(allocaOp, name)
	}

	// Return the address.
	return resultOf(allocaOp).AsValue()
}

func (b *Builder) emitConstBool(ctx context.Context, value bool, T mlir.TypeLike, location mlir.LocationLike) mlir.Value {
	op := goir.NewConstantOperation(b.ctx, b.boolAttr(value), nil, T, location)
	appendOperation(ctx, op)
	return resultOf(op).AsValue()
}

func (b *Builder) emitConstComplex64(ctx context.Context, r float32, i float32, T mlir.TypeLike, location mlir.LocationLike) mlir.Value {
	attr := goir.NewComplexAttr(b.ctx, b.f32, float64(r), float64(i))
	op := goir.NewConstantOperation(b.ctx, attr, nil, T, location)
	appendOperation(ctx, op)
	return resultOf(op).AsValue()
}

func (b *Builder) emitConstComplex128(ctx context.Context, r float64, i float64, T mlir.TypeLike, location mlir.LocationLike) mlir.Value {
	attr := goir.NewComplexAttr(b.ctx, b.f64, r, i)
	op := goir.NewConstantOperation(b.ctx, attr, nil, T, location)
	appendOperation(ctx, op)
	return resultOf(op).AsValue()
}

func (b *Builder) emitConstFloat32(ctx context.Context, value float32, T mlir.TypeLike, location mlir.LocationLike) mlir.Value {
	op := goir.NewConstantOperation(b.ctx, mlir.NewFloatAttr(b.ctx, b.f32, float64(value)), nil, T, location)
	appendOperation(ctx, op)
	return resultOf(op).AsValue()
}

func (b *Builder) emitConstFloat64(ctx context.Context, value float64, T mlir.TypeLike, location mlir.LocationLike) mlir.Value {
	op := goir.NewConstantOperation(b.ctx, mlir.NewFloatAttr(b.ctx, b.f64, value), nil, T, location)
	appendOperation(ctx, op)
	return resultOf(op).AsValue()
}

func (b *Builder) emitConstInt(ctx context.Context, value int64, T mlir.TypeLike, location mlir.LocationLike) mlir.Value {
	// NOTE: The integer type used with integer attributes must be signless.
	op := goir.NewConstantOperation(b.ctx, b.intAttr(value), nil, T, location)
	appendOperation(ctx, op)
	return resultOf(op).AsValue()
}

func (b *Builder) emitConstString(ctx context.Context, value string, T mlir.TypeLike, location mlir.LocationLike) mlir.Value {
	return b.emitConstStringInSection(ctx, value, T, "", location)
}

func (b *Builder) emitConstStringInSection(ctx context.Context, value string, T mlir.TypeLike, section string, location mlir.LocationLike) mlir.Value {
	op := goir.NewConstantOperation(b.ctx, mlir.NewStringAttr(b.ctx, value), nil, T, location)
	if len(section) > 0 {
		op.SetAttributeByName("go.section", mlir.NewStringAttr(b.ctx, section))
	}
	appendOperation(ctx, op)
	return resultOf(op).AsValue()
}

func (b *Builder) emitStringValue(ctx context.Context, arr mlir.ValueLike, length mlir.ValueLike, location mlir.LocationLike) mlir.Value {
	// Create the zero value of the string runtime type.
	zeroOp := goir.NewZeroOperation(b.ctx, b._string, location)
	appendOperation(ctx, zeroOp)

	// Build the string struct.
	insertOp := goir.NewInsertOperation(b.ctx, 0, arr, resultOf(zeroOp), b._string, location)
	appendOperation(ctx, insertOp)
	insertOp = goir.NewInsertOperation(b.ctx, 1, length, resultOf(insertOp), b._string, location)
	appendOperation(ctx, insertOp)
	return resultOf(insertOp).AsValue()
}

func (b *Builder) emitConstSlice(ctx context.Context, arr mlir.ValueLike, length int, location mlir.LocationLike) mlir.Value {
	// Create the zero value of the slice runtime type.
	zeroOp := goir.NewZeroOperation(b.ctx, b._slice, location)
	appendOperation(ctx, zeroOp)

	// Create the constant length value.
	constLen := b.emitConstInt(ctx, int64(length), b.si, location)

	// Build the slice struct.
	insertOp := goir.NewInsertOperation(b.ctx, 0, arr, resultOf(zeroOp), b._slice, location)
	appendOperation(ctx, insertOp)
	insertOp = goir.NewInsertOperation(b.ctx, 1, constLen, resultOf(insertOp), b._slice, location)
	appendOperation(ctx, insertOp)
	insertOp = goir.NewInsertOperation(b.ctx, 2, constLen, resultOf(insertOp), b._slice, location)
	appendOperation(ctx, insertOp)
	return resultOf(insertOp).AsValue()
}

func (b *Builder) emitEmbedSlice(ctx context.Context, data []byte, T mlir.TypeLike, section string, location mlir.LocationLike) mlir.Value {
	// Emit the data as a string constant (reuses the existing GlobalConstantsPass pipeline
	// which materializes string data into LLVM globals).
	strVal := b.emitConstStringInSection(ctx, string(data), b.str, section, location)
	rawStrVal := b.bitcastTo(ctx, strVal, b._string, location)

	// Extract the pointer from the string struct (index 0).
	extractOp := goir.NewExtractOperation(b.ctx, 0, b.ptr, rawStrVal, location)
	appendOperation(ctx, extractOp)
	arr := resultOf(extractOp).AsValue()

	// Create the constant length value.
	constLen := b.emitConstInt(ctx, int64(len(data)), b.si, location)

	// Build the slice struct using the actual Go slice type (e.g., !go.slice<!go.ui8>).
	zeroOp := goir.NewZeroOperation(b.ctx, b._slice, location)
	appendOperation(ctx, zeroOp)
	insertOp := goir.NewInsertOperation(b.ctx, 0, arr, resultOf(zeroOp), b._slice, location)
	appendOperation(ctx, insertOp)
	insertOp = goir.NewInsertOperation(b.ctx, 1, constLen, resultOf(insertOp), b._slice, location)
	appendOperation(ctx, insertOp)
	insertOp = goir.NewInsertOperation(b.ctx, 2, constLen, resultOf(insertOp), b._slice, location)
	appendOperation(ctx, insertOp)
	rawSlice := resultOf(insertOp).AsValue()

	// Bitcast to the expected slice type.
	return b.bitcastTo(ctx, rawSlice, T, location)
}

func (b *Builder) emitAsset(
	ctx context.Context,
	data []byte,
	T mlir.TypeLike,
	section string,
	location mlir.LocationLike,
) mlir.Value {
	// Materialize the encoded SGFX bytes using the same path as embedded data.
	strVal := b.emitConstStringInSection(ctx, string(data), b.str, section, location)
	rawStrVal := b.bitcastTo(ctx, strVal, b._string, location)

	// Extract string backing pointer.
	extractOp := goir.NewExtractOperation(b.ctx, 0, b.ptr, rawStrVal, location)
	appendOperation(ctx, extractOp)

	// Bitcast the string backing pointer to the asset header pointer type.
	hptr := b.bitcastTo(ctx, resultOf(extractOp).AsValue(), b._assetHeaderPtr, location)

	// Advance the backing pointer to the start of the bitmap data.
	constHeaderLenVal := b.emitConstInt(ctx, asset.HeaderSize, b.uiptr, location)
	bptrOp := goir.NewPtrToIntOperation(b.ctx, resultOf(extractOp).AsValue(), b.uiptr, location)
	appendOperation(ctx, bptrOp)
	bptrOp = goir.NewAddIOperation(b.ctx, b.uiptr, resultOf(bptrOp), constHeaderLenVal, location)
	appendOperation(ctx, bptrOp)
	bptrOp = goir.NewIntToPtrOperation(b.ctx, resultOf(bptrOp), b.ptr, location)
	appendOperation(ctx, bptrOp)
	bptr := resultOf(bptrOp).AsValue()

	// Construct and return the asset reference type value.
	zeroOp := goir.NewZeroOperation(b.ctx, b._asset, location)
	appendOperation(ctx, zeroOp)
	insertOp := goir.NewInsertOperation(b.ctx, 0, hptr, resultOf(zeroOp), b._asset, location)
	appendOperation(ctx, insertOp)
	insertOp = goir.NewInsertOperation(b.ctx, 1, bptr, resultOf(insertOp), b._asset, location)
	appendOperation(ctx, insertOp)
	return resultOf(insertOp).AsValue()
}

func (b *Builder) emitZeroValue(ctx context.Context, T types.Type, location mlir.LocationLike) mlir.Value {
	zeroOp := goir.NewZeroOperation(b.ctx, b.GetStoredType(ctx, T), location)
	appendOperation(ctx, zeroOp)
	return resultOf(zeroOp).AsValue()
}

func (b *Builder) emitInterfaceValue(ctx context.Context, T types.Type, valueType types.Type, value mlir.ValueLike, location mlir.LocationLike) mlir.Value {
	var addr mlir.Value
	if isPointer(valueType) {
		// Use the pointer value directly.
		addr = value.AsValue()
	} else {
		// Copy the value onto the stack.
		// NOTE: This value may escape to the heap later.
		addr = b.makeCopyOf(ctx, value, valueType, location)
	}

	// Generate methods for named types.
	var namedType *types.Named
	switch valueType := valueType.(type) {
	case *types.Named:
		namedType = valueType
	case *types.Pointer:
		if T, ok := valueType.Elem().(*types.Named); ok {
			namedType = T
		}
	}

	var generate func(ctx context.Context, namedType *types.Named)
	generate = func(ctx context.Context, namedType *types.Named) {
		// Queue the methods of this named type to be generated.
		b.queueNamedTypeJobs(ctx, namedType)

		// Examine the named type for any embedded types if it's a struct type.
		if structType, ok := types.Unalias(namedType.Underlying()).(*types.Struct); ok {
			for field := range structType.Fields() {
				if field.Embedded() {
					if embeddedType, ok := types.Unalias(field.Type()).(*types.Named); ok {
						// Generate the methods of the embedded named type.
						generate(ctx, embeddedType)
					}
				}
			}
		}
	}

	if namedType != nil {
		generate(ctx, namedType)
	}

	// Create the interface value.
	interfaceT := b.GetType(ctx, T)
	dynamicT := b.GetType(ctx, valueType)
	makeOp := goir.NewMakeInterfaceOperation(b.ctx, interfaceT, dynamicT, addr, location)
	appendOperation(ctx, makeOp)
	return resultOf(makeOp).AsValue()
}

func (b *Builder) emitChangeType(ctx context.Context, T types.Type, value mlir.ValueLike, location mlir.LocationLike) mlir.Value {
	interfaceT := b.GetType(ctx, T)
	changeOp := goir.NewChangeInterfaceOperation(b.ctx, value, interfaceT, location)
	appendOperation(ctx, changeOp)
	return resultOf(changeOp).AsValue()
}

func (b *Builder) bitcastTo(ctx context.Context, X mlir.ValueLike, T mlir.TypeLike, location mlir.LocationLike) mlir.Value {
	bitcastOp := goir.NewBitcastOperation(b.ctx, X, T, location)
	appendOperation(ctx, bitcastOp)
	return resultOf(bitcastOp).AsValue()
}

func (b *Builder) addressOfSymbol(ctx context.Context, symbol string, T mlir.TypeLike, location mlir.LocationLike) mlir.Value {
	addressOfOp := goir.NewAddressOfOperation(b.ctx, symbol, T, location)
	appendOperation(ctx, addressOfOp)
	return resultOf(addressOfOp).AsValue()
}

func (b *Builder) addressOf(ctx context.Context, expr ast.Expr, location mlir.LocationLike) mlir.Value {
	switch expr := expr.(type) {
	case *ast.Ident:
		// Return the address of the original allocation for the value.
		return b.valueOf(ctx, expr).Pointer(ctx, location)
	case *ast.IndexExpr:
		return b.emitIndexAddr(ctx, expr)
	case *ast.SelectorExpr:
		return b.emitSelectAddr(ctx, expr)
	default:
		// Load the value.
		value := b.emitExpr(ctx, expr)[0].AsValue()

		// Create a reference to the loaded value.
		return b.makeCopyOf(ctx, value, b.typeOf(ctx, expr), location)
	}
}

func (b *Builder) exprTypes(ctx context.Context, expr ...ast.Expr) []mlir.TypeLike {
	var result []mlir.TypeLike
	for _, expr := range expr {
		switch t := b.typeOf(ctx, expr).(type) {
		case *types.Tuple:
			for i := range t.Len() {
				result = append(result, b.GetStoredType(ctx, t.At(i).Type()))
			}
		default:
			result = append(result, b.GetStoredType(ctx, t))
		}
	}
	return result
}

func (b *Builder) exprValues(ctx context.Context, expr ...ast.Expr) []mlir.ValueLike {
	var result []mlir.ValueLike
	for _, expr := range expr {
		result = append(result, b.emitExpr(ctx, expr)...)
	}
	return result
}

func (b *Builder) types(T ...mlir.TypeLike) []mlir.TypeLike {
	return T
}

func (b *Builder) values(value ...mlir.ValueLike) []mlir.ValueLike {
	return value
}

func (b *Builder) locations(locs ...mlir.LocationLike) []mlir.LocationLike {
	return locs
}

func (b *Builder) emitLoad(ctx context.Context, value mlir.ValueLike, typ mlir.TypeLike, location mlir.LocationLike) mlir.Value {
	// Emit a nil pointer check operation before the load.
	nilCheckOp := goir.NewNilPointerCheckOperation(b.ctx, value, location)
	appendOperation(ctx, nilCheckOp)

	// Emit the load operation.
	loadOp := goir.NewLoadOperation(b.ctx, value, typ, location)
	appendOperation(ctx, loadOp)
	return resultOf(loadOp).AsValue()
}

func (b *Builder) emitStore(ctx context.Context, value mlir.ValueLike, address mlir.ValueLike, location mlir.LocationLike) {
	// Emit a nil pointer check operation before the store.
	nilCheckOp := goir.NewNilPointerCheckOperation(b.ctx, address, location)
	appendOperation(ctx, nilCheckOp)

	// Emit the store operation.
	storeOp := goir.NewStoreOperation(b.ctx, value, address, location)
	appendOperation(ctx, storeOp)
}
