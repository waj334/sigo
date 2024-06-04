package ssa

import (
	"context"
	"go/ast"
	"go/types"
	"omibyte.io/sigo/mlir"
)

func (b *Builder) valueOf(ctx context.Context, node ast.Node) Value {
	switch node := node.(type) {
	case *ast.SelectorExpr:
		return b.NewTempValue(b.emitSelectAddr(ctx, node))
	case *ast.IndexExpr:
		return b.NewTempValue(b.emitIndexAddr(ctx, node))
	case *ast.Ident:
		// Look in the current function's locals first.
		if data := currentFuncData(ctx); data != nil {
			data.mutex.RLock()
			obj := b.objectOf(ctx, node)
			if obj == nil {
				panic("object is nil")
			}
			if value, ok := data.locals[obj]; ok {
				data.mutex.RUnlock()
				return value
			}
			data.mutex.RUnlock()
		}

		// Look up the value by object.
		obj := b.objectOf(ctx, node)
		return b.lookupValue(obj)
	default:
		return nil
	}
}

func (b *Builder) lookupValue(obj types.Object) Value {
	// Lock the value cache mutex for reading.
	b.valueCacheMutex.RLock()
	defer b.valueCacheMutex.RUnlock()
	return b.valueCache[obj]
}

func (b *Builder) emitLocalVar(ctx context.Context, obj types.Object, T mlir.Type) *LocalValue {
	// Allocate memory for this local variable on the stack.
	// NOTE: It may be determined later that this variable escapes to the heap and the following operation will be
	//       replaced by a heap allocation.

	ptrType := mlir.GoCreatePointerType(T)
	allocaOp := mlir.GoCreateAllocaOperation(b.config.Ctx, ptrType, T, nil, false, b.location(obj.Pos()))

	// NOTE: Omitted identifiers ( `_` )  will not have any debug information attached.
	if len(obj.Name()) > 0 && obj.Name() != "_" {
		mlir.GoAllocaOperationSetName(allocaOp, obj.Name())
	}
	appendOperation(ctx, allocaOp)
	value := &LocalValue{
		ptr: resultOf(allocaOp),
		T:   T,
		b:   b,
	}

	b.valueCacheMutex.Lock()
	defer b.valueCacheMutex.Unlock()
	b.valueCache[obj] = value
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
	var linkage mlir.Attribute
	if obj.Exported() || info.Exported || len(info.LinkName) > 0 {
		linkage = mlir.GetLLVMLinkageAttr(b.ctx, "external")
	} else {
		linkage = mlir.GetLLVMLinkageAttr(b.ctx, "private")
	}

	// Fuse the location with the compile unit if applicable.
	location := b.location(obj.Pos())
	if file := b.config.Fset.File(obj.Pos()); file != nil {
		if compileUnitAttr, ok := b.compileUnits[file]; ok {
			location = mlir.LocationFusedGet(b.ctx, []mlir.Location{location}, compileUnitAttr)
		}
	}

	// Emit the global variable.
	globalOp := mlir.GoCreateGlobalOperation(b.ctx, linkage, symbol, T, location)
	b.appendToModule(globalOp)
	value := &GlobalValue{
		symbol: symbol,
		T:      T,
		ctx:    b.ctx,
	}

	b.valueCacheMutex.Lock()
	defer b.valueCacheMutex.Unlock()

	b.valueCache[obj] = value
	b.addSymbol(globalOp)

	return value
}

func (b *Builder) emitCastPointerToInt(ctx context.Context, X mlir.Value, location mlir.Location) mlir.Value {
	op := mlir.GoCreatePtrToIntOperation(b.ctx, X, b.GetStoredType(ctx, types.Typ[types.Uintptr]), location)
	appendOperation(ctx, op)
	return resultOf(op)
}

func (b *Builder) makeCopyOf(ctx context.Context, X mlir.Value, location mlir.Location) mlir.Value {
	elementType := mlir.ValueGetType(X)
	ptrType := mlir.GoCreatePointerType(elementType)

	// Taking the address in global initializers causes the allocation to escape to the heap.
	isHeap := false
	if isGlobalContext(ctx) {
		isHeap = true
	}

	// Allocate memory on the stack to hold the object.
	allocaOp := mlir.GoCreateAllocaOperation(b.ctx, ptrType, elementType, nil, isHeap, location)
	appendOperation(ctx, allocaOp)

	// Store the object at the address.
	storeOp := mlir.GoCreateStoreOperation(b.ctx, X, resultOf(allocaOp), location)
	appendOperation(ctx, storeOp)

	// Return the address.
	return resultOf(allocaOp)
}

func (b *Builder) emitConstBool(ctx context.Context, value bool, location mlir.Location) mlir.Value {
	var v int64 = 0
	if value {
		v = 1
	}
	op := mlir.GoCreateConstantOperation(b.ctx, mlir.IntegerAttrGet(b.i1, v), b.i1, location)
	appendOperation(ctx, op)
	return resultOf(op)
}

func (b *Builder) emitConstComplex64(ctx context.Context, r float32, i float32, T mlir.Type, location mlir.Location) mlir.Value {
	attr := mlir.GoCreateComplexNumberAttr(b.ctx, T, float64(r), float64(i))
	op := mlir.GoCreateConstantOperation(b.ctx, attr, T, location)
	appendOperation(ctx, op)
	return resultOf(op)
}

func (b *Builder) emitConstComplex128(ctx context.Context, r float64, i float64, T mlir.Type, location mlir.Location) mlir.Value {
	attr := mlir.GoCreateComplexNumberAttr(b.ctx, T, r, i)
	op := mlir.GoCreateConstantOperation(b.ctx, attr, T, location)
	appendOperation(ctx, op)
	return resultOf(op)
}

func (b *Builder) emitConstFloat32(ctx context.Context, value float32, location mlir.Location) mlir.Value {
	op := mlir.GoCreateConstantOperation(b.ctx, mlir.FloatAttrDoubleGet(b.ctx, b.f32, float64(value)), b.f32, location)
	appendOperation(ctx, op)
	return resultOf(op)
}

func (b *Builder) emitConstFloat64(ctx context.Context, value float64, location mlir.Location) mlir.Value {
	op := mlir.GoCreateConstantOperation(b.ctx, mlir.FloatAttrDoubleGet(b.ctx, b.f64, value), b.f64, location)
	appendOperation(ctx, op)
	return resultOf(op)
}

func (b *Builder) emitConstInt(ctx context.Context, value int64, T mlir.Type, location mlir.Location) mlir.Value {
	// NOTE: The integer type used with integer attributes must be signless.
	attrIntType := mlir.IntegerTypeGet(b.ctx, 64)
	op := mlir.GoCreateConstantOperation(b.ctx, mlir.IntegerAttrGet(attrIntType, value), T, location)
	appendOperation(ctx, op)
	return resultOf(op)
}

func (b *Builder) emitConstString(ctx context.Context, value string, location mlir.Location) mlir.Value {
	op := mlir.GoCreateConstantOperation(b.ctx, mlir.StringAttrGet(b.ctx, value), b.str, location)
	appendOperation(ctx, op)
	return resultOf(op)
}

func (b *Builder) emitStringValue(ctx context.Context, arr mlir.Value, length mlir.Value, location mlir.Location) mlir.Value {
	// Create the zero value of the string runtime type.
	zeroOp := mlir.GoCreateZeroOperation(b.ctx, b._string, location)
	appendOperation(ctx, zeroOp)

	// Build the string struct.
	insertOp := mlir.GoCreateInsertOperation(b.ctx, 0, arr, resultOf(zeroOp), b._string, location)
	appendOperation(ctx, insertOp)
	insertOp = mlir.GoCreateInsertOperation(b.ctx, 1, length, resultOf(insertOp), b._string, location)
	appendOperation(ctx, insertOp)
	return resultOf(insertOp)
}

func (b *Builder) emitConstSlice(ctx context.Context, arr mlir.Value, length int, location mlir.Location) mlir.Value {
	// Create the zero value of the slice runtime type.
	zeroOp := mlir.GoCreateZeroOperation(b.ctx, b._slice, location)
	appendOperation(ctx, zeroOp)

	// Create the constant length value.
	constLen := b.emitConstInt(ctx, int64(length), b.si, location)

	// Build the slice struct.
	insertOp := mlir.GoCreateInsertOperation(b.ctx, 0, arr, resultOf(zeroOp), b._slice, location)
	appendOperation(ctx, insertOp)
	insertOp = mlir.GoCreateInsertOperation(b.ctx, 1, constLen, resultOf(insertOp), b._slice, location)
	appendOperation(ctx, insertOp)
	insertOp = mlir.GoCreateInsertOperation(b.ctx, 2, constLen, resultOf(insertOp), b._slice, location)
	appendOperation(ctx, insertOp)
	return resultOf(insertOp)
}

func (b *Builder) emitZeroValue(ctx context.Context, T types.Type, location mlir.Location) mlir.Value {
	zeroOp := mlir.GoCreateZeroOperation(b.ctx, b.GetStoredType(ctx, T), location)
	appendOperation(ctx, zeroOp)
	return resultOf(zeroOp)
}

func (b *Builder) emitInterfaceValue(ctx context.Context, T types.Type, value mlir.Value, location mlir.Location) mlir.Value {
	// Copy the value onto the stack.
	// NOTE: This value may escape to the heap later.
	addr := b.makeCopyOf(ctx, value, location)

	// Create the interface value.
	interfaceT := b.GetType(ctx, T)
	dynamicT := mlir.ValueGetType(value)
	makeOp := mlir.GoCreateMakeInterfaceOperation(b.ctx, interfaceT, dynamicT, addr, location)
	appendOperation(ctx, makeOp)
	return resultOf(makeOp)
}

func (b *Builder) emitChangeType(ctx context.Context, T types.Type, value mlir.Value, location mlir.Location) mlir.Value {
	interfaceT := b.GetType(ctx, T)
	changeOp := mlir.GoCreateChangeInterfaceOperation(b.ctx, value, interfaceT, location)
	appendOperation(ctx, changeOp)
	return resultOf(changeOp)
}

func (b *Builder) bitcastTo(ctx context.Context, X mlir.Value, T mlir.Type, location mlir.Location) mlir.Value {
	bitcastOp := mlir.GoCreateBitcastOperation(b.ctx, X, T, location)
	appendOperation(ctx, bitcastOp)
	return resultOf(bitcastOp)
}

func (b *Builder) addressOfSymbol(ctx context.Context, symbol string, T mlir.Type, location mlir.Location) mlir.Value {
	addressOfOp := mlir.GoCreateAddressOfOperation(b.ctx, symbol, T, location)
	appendOperation(ctx, addressOfOp)
	return resultOf(addressOfOp)
}

func (b *Builder) addressOf(ctx context.Context, expr ast.Expr, location mlir.Location) mlir.Value {
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
		value := b.emitExpr(ctx, expr)[0]

		// Create a reference to the loaded value.
		return b.makeCopyOf(ctx, value, location)
	}
}

func (b *Builder) exprTypes(ctx context.Context, expr ...ast.Expr) []mlir.Type {
	var result []mlir.Type
	for _, expr := range expr {
		result = append(result, b.GetStoredType(ctx, b.typeOf(ctx, expr)))
	}
	return result
}

func (b *Builder) exprValues(ctx context.Context, expr ...ast.Expr) []mlir.Value {
	var result []mlir.Value
	for _, expr := range expr {
		result = append(result, b.emitExpr(ctx, expr)...)
	}
	return result
}

func (b *Builder) typeInfoOf(ctx context.Context, T types.Type, location mlir.Location) mlir.Value {
	op := mlir.GoCreateTypeInfoOperation(b.ctx, b.typeInfoPtr, b.GetType(ctx, T), location)
	appendOperation(ctx, op)
	return resultOf(op)
}

func (b *Builder) types(T ...mlir.Type) []mlir.Type {
	return T
}

func (b *Builder) values(value ...mlir.Value) []mlir.Value {
	return value
}
