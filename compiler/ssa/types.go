package ssa

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"hash/fnv"

	"omibyte.io/sigo/mlir"
)

type typeCacheNestedLockKey struct{}

func (b *Builder) GetType(ctx context.Context, T types.Type) (result mlir.Type) {

	if typeHasFlags(T, types.IsUntyped) {
		panic("Untyped type not allowed")
	}

	// NOTE: The anonymous function usage below exists for making handling the read lock easier.
	if func() bool {
		// Lock the type cache for reading while it is accessed if no recursive lock is currently held.
		isLockNested := ctx.Value(typeCacheNestedLockKey{})
		if isLockNested == nil || !isLockNested.(bool) {
			// Lock the type cache mutex for reading.
			b.typeCacheMutex.RLock()
			defer b.typeCacheMutex.RUnlock()
		}

		// Look up the previously generated type for the input type.
		var ok bool
		result, ok = b.typeCache[T]
		return ok
	}() {
		return
	}

	// Handle recursive write lock.
	isLockNested := ctx.Value(typeCacheNestedLockKey{})
	if isLockNested == nil || !isLockNested.(bool) {
		// Lock the type cache mutex
		b.typeCacheMutex.Lock()
		defer b.typeCacheMutex.Unlock()

		// All nested calls to this function do not need to lock.
		ctx = context.WithValue(ctx, typeCacheNestedLockKey{}, true)
	}

	switch T := T.(type) {
	case *types.Array:
		result = b.createArrayType(ctx, T)
	case *types.Basic:
		result = b.createBasicType(ctx, T)
		if typeHasFlags(T, types.IsUntyped) {
			// Do not cache untyped types.
			return result
		}
	case *types.Chan:
		result = b.createChanType(ctx, T)
	case *types.Interface:
		result = b.createInterfaceType(ctx, T)
	case *types.Map:
		result = b.createMapType(ctx, T)
	case *types.Named:
		result = b.createNamedType(ctx, T)
	case *types.Pointer:
		result = b.createPointerType(ctx, T)
	case *types.Signature:
		result = b.createSignatureType(ctx, T, false)
	case *types.Slice:
		result = b.createSliceType(ctx, T)
	case *types.Struct:
		result = b.createStructType(ctx, T)
	case *types.Tuple:
		println(T.String())
		panic("unreachable")
	case *types.TypeParam:
		// Look up the instantiated type in the data of the current function.
		data := currentFuncData(ctx)

		// Lock the function data for reading while it is accessed below
		data.mutex.RLock()
		defer data.mutex.RUnlock()

		// Skip the type cache by returning now.
		return b.GetType(ctx, data.typeMap[T.Index()])
	default:
		panic("unhandled type")
	}

	if result == nil {
		panic("no type was created")
	}

	b.typeCache[T] = result
	return result
}

func (b *Builder) createTypeDeclaration(ctx context.Context, T types.Type, pos token.Pos) {
	if _, ok := b.declaredTypes[T]; ok {
		// Do not create the operation more than once for an individual type.
		return
	}

	// Mark the type now to prevent infinite recursion with recursive types.
	b.declaredTypes[T] = struct{}{}

	location := b.location(pos)
	var extraData []mlir.NamedAttribute

	switch T := T.(type) {
	case *types.Array:
		// Create the data for the element type.
		b.createTypeDeclaration(ctx, T.Elem(), pos)
	case *types.Basic:

	case *types.Chan:
		// Create the data for the element type.
		b.createTypeDeclaration(ctx, T.Elem(), pos)
	case *types.Interface:

	case *types.Map:
		// Create the data for the key type.
		b.createTypeDeclaration(ctx, T.Key(), pos)

		// Create the data for the element type.
		b.createTypeDeclaration(ctx, T.Elem(), pos)
	case *types.Named:
		// Create the methods dictionary if this named type has methods.
		var methodSymbols mlir.Attribute
		if T.NumMethods() > 0 {
			entries := make([]mlir.Attribute, T.NumMethods())
			for i := 0; i < T.NumMethods(); i++ {
				method := T.Method(i)
				symbol := mangleSymbol(qualifiedFuncName(method))
				refAttr := mlir.FlatSymbolRefAttrGet(b.ctx, symbol)
				entries[i] = refAttr

				// Create type information for the signature.
				b.createTypeDeclaration(ctx, method.Type(), pos)
			}
			methodSymbols = mlir.ArrayAttrGet(b.ctx, entries)
			extraData = append(extraData, b.namedOf("methods", methodSymbols))
		}
		// Create the data for the underlying type.
		b.createTypeDeclaration(ctx, T.Underlying(), pos)
	case *types.Pointer:
		// Create the data for the element type.
		b.createTypeDeclaration(ctx, T.Elem(), pos)
	case *types.Signature:
		if T.Recv() != nil {
			extraData = append(extraData, b.namedOf("receiver", mlir.BoolAttrGet(b.ctx, 1)))
		} else {
			extraData = append(extraData, b.namedOf("receiver", mlir.BoolAttrGet(b.ctx, 0)))
		}
	case *types.Slice:
		// Create the data for the element type.
		b.createTypeDeclaration(ctx, T.Elem(), pos)
	case *types.Struct:
		fields := make([]mlir.Attribute, T.NumFields())
		tags := make([]mlir.Attribute, T.NumFields())
		for i := 0; i < T.NumFields(); i++ {
			field := T.Field(i)
			fields[i] = mlir.StringAttrGet(b.ctx, field.Name())
			tags[i] = mlir.StringAttrGet(b.ctx, T.Tag(i))

			b.createTypeDeclaration(ctx, field.Type(), pos)
		}
		extraData = append(extraData, b.namedOf("fields", mlir.ArrayAttrGet(b.ctx, fields)))
		extraData = append(extraData, b.namedOf("tags", mlir.ArrayAttrGet(b.ctx, tags)))
	case *types.Tuple:
		panic("unreachable")
	case *types.TypeParam:

	default:
		panic("unhandled type")
	}

	// Fuse the location with the compile unit if applicable.
	if file := b.config.Fset.File(pos); file != nil {
		if compileUnitAttr, ok := b.compileUnits[file]; ok {
			location = mlir.LocationFusedGet(b.ctx, []mlir.Location{location}, compileUnitAttr)
		}
	}

	// Declare this type.
	extraDataDict := mlir.DictionaryAttrGet(b.ctx, extraData)
	declareOp := mlir.GoCreateDeclareTypeOperation(b.ctx, b.GetType(ctx, T), extraDataDict, location)
	b.appendToModule(declareOp)
}

func (b *Builder) createArrayType(ctx context.Context, T *types.Array) mlir.Type {
	// Create the element type.
	elementType := b.GetStoredType(ctx, T.Elem())

	// Return the array type.
	return mlir.GoCreateArrayType(elementType, int(T.Len()))
}

func (b *Builder) createBasicType(ctx context.Context, T *types.Basic) mlir.Type {
	switch T.Kind() {
	case types.Bool:
		return mlir.GoCreateBooleanType(b.ctx)
	case types.Int:
		return mlir.GoCreateSignedIntType(b.ctx, 0)
	case types.Uint:
		return mlir.GoCreateUnsignedIntType(b.ctx, 0)
	case types.Uintptr:
		return mlir.GoCreateUintptrType(b.ctx)
	case types.Int8:
		return mlir.GoCreateSignedIntType(b.ctx, 8)
	case types.Uint8:
		return mlir.GoCreateUnsignedIntType(b.ctx, 8)
	case types.Int16:
		return mlir.GoCreateSignedIntType(b.ctx, 16)
	case types.Uint16:
		return mlir.GoCreateUnsignedIntType(b.ctx, 16)
	case types.Int32:
		return mlir.GoCreateSignedIntType(b.ctx, 32)
	case types.Uint32:
		return mlir.GoCreateUnsignedIntType(b.ctx, 32)
	case types.Int64:
		return mlir.GoCreateSignedIntType(b.ctx, 64)
	case types.Uint64:
		return mlir.GoCreateUnsignedIntType(b.ctx, 64)
	case types.Float32:
		return mlir.F32TypeGet(b.ctx)
	case types.Float64:
		return mlir.F64TypeGet(b.ctx)
	case types.Complex64:
		return mlir.ComplexTypeGet(mlir.F32TypeGet(b.ctx))
	case types.Complex128:
		return mlir.ComplexTypeGet(mlir.F64TypeGet(b.ctx))
	case types.String:
		return mlir.GoCreateStringType(b.ctx)
	case types.UnsafePointer:
		return mlir.GoCreateUnsafePointerType(b.ctx)
	default:
		panic(fmt.Sprintf("unknown basic type %+v", T.Kind()))
	}
}

func (b *Builder) createChanType(ctx context.Context, T *types.Chan) mlir.Type {
	// Create the element type.
	elementType := b.GetStoredType(ctx, T.Elem())

	// Return the chan type
	switch T.Dir() {
	case types.SendRecv:
		return mlir.GoCreateChanType(elementType, mlir.GoChanDirection_SendRecv)
	case types.SendOnly:
		return mlir.GoCreateChanType(elementType, mlir.GoChanDirection_SendOnly)
	case types.RecvOnly:
		return mlir.GoCreateChanType(elementType, mlir.GoChanDirection_RecvOnly)
	default:
		panic("invalid chan direction")
	}
}

func (b *Builder) createInterfaceType(ctx context.Context, T *types.Interface) mlir.Type {
	var methodNames []string
	var methods []mlir.Type

	// Get the identifier if this is actually a named interface type.
	identifier := currentIdentifier(ctx)

	// Unset the name in a new context.
	ctx = context.WithValue(ctx, identifierKey{}, "")

	// NOTE: Named interfaces need to declare the interface type first before creating the methods in order to prevent
	//       infinite recursion.
	if len(identifier) > 0 {
		// Create a named interface.
		result := mlir.GoCreateNamedInterfaceType(b.ctx, identifier)

		// Prevent infinite recursion when mutually recursive types are encountered.
		b.typeCache[T] = result

		// Create the function signatures.
		for i := 0; i < T.NumMethods(); i++ {
			method := T.Method(i)
			MT := b.createSignatureType(ctx, method.Type().(*types.Signature), true)
			methodNames = append(methodNames, method.Name())
			methods = append(methods, MT)
		}

		// Set the interface methods.
		mlir.GoSetNamedInterfaceMethods(b.ctx, result, methodNames, methods)

		// Return the interface type.
		return result
	} else {
		// Create the function signatures
		for i := 0; i < T.NumMethods(); i++ {
			method := T.Method(i)
			MT := b.createSignatureType(ctx, method.Type().(*types.Signature), true)
			methodNames = append(methodNames, method.Name())
			methods = append(methods, MT)
		}

		// Create a literal interface
		return mlir.GoCreateInterfaceType(b.ctx, methodNames, methods)
	}
}

func (b *Builder) createNamedType(ctx context.Context, T *types.Named) mlir.Type {
	// Format the qualified identifier for this type with respect to its origin package.
	identifier := qualifiedName(T.Obj().Name(), T.Obj().Pkg())

	if T.TypeArgs().Len() > 0 {
		hasher := fnv.New32()
		hasher.Write([]byte(T.Obj().Name()))
		for i := 0; i < T.TypeArgs().Len(); i++ {
			hasher.Write([]byte(T.TypeArgs().At(i).String()))
		}
		hasher.Sum32()
		identifier += fmt.Sprintf("$%X", hasher.Sum32())
	}

	// Add the identifier to the current context.
	ctx = newContextWithIdentifier(ctx, identifier)

	// Create the underlying type.
	underlyingType := b.GetType(ctx, T.Underlying())

	// Collect method symbols.
	entries := make([]mlir.Attribute, T.NumMethods())
	for i := 0; i < T.NumMethods(); i++ {
		method := T.Method(i)
		symbol := mangleSymbol(qualifiedFuncName(method))
		refAttr := mlir.FlatSymbolRefAttrGet(b.ctx, symbol)
		entries[i] = refAttr
	}
	methodSymbols := mlir.ArrayAttrGet(b.ctx, entries)

	// Create the named type now.
	result := mlir.GoCreateNamedType(underlyingType, identifier, methodSymbols)

	// Prevent infinite recursion within metadata and mutually recursive types by mapping the named type now.
	b.typeCache[T] = result

	return result
}

func (b *Builder) createMapType(ctx context.Context, T *types.Map) mlir.Type {
	// Create the key type.
	keyType := b.GetStoredType(ctx, T.Key())

	// Create the element type.
	elementType := b.GetStoredType(ctx, T.Elem())

	// Return the map type.
	return mlir.GoCreateMapType(keyType, elementType)
}

func (b *Builder) createPointerType(ctx context.Context, T *types.Pointer) mlir.Type {
	elementType := b.GetStoredType(ctx, T.Elem())
	return mlir.GoCreatePointerType(elementType)
}

func (b *Builder) pointerOf(ctx context.Context, T types.Type) mlir.Type {
	ptrType := types.NewPointer(T)
	return b.GetStoredType(ctx, ptrType)
}

func (b *Builder) funcPointerOf(ctx context.Context, T *types.Signature) mlir.Type {
	fnT := b.GetType(ctx, T)
	return mlir.GoCreatePointerType(fnT)
}

func (b *Builder) createSliceType(ctx context.Context, T *types.Slice) mlir.Type {
	// Create the element type
	elementType := b.GetStoredType(ctx, T.Elem())

	// Create the slice type
	return mlir.GoCreateSliceType(elementType)
}

func (b *Builder) createSignatureType(ctx context.Context, T *types.Signature, isInterface bool) mlir.Type {
	var receiver mlir.Type
	var inputs []mlir.Type
	var results []mlir.Type

	if T.Recv() != nil {
		// The receiver is always the first parameter to a method.
		receiver = b.GetStoredType(ctx, T.Recv().Type())
	}

	for i := 0; i < T.Params().Len(); i++ {
		inputs = append(inputs, b.GetStoredType(ctx, T.Params().At(i).Type()))
	}

	for i := 0; i < T.Results().Len(); i++ {
		results = append(results, b.GetStoredType(ctx, T.Results().At(i).Type()))
	}

	return mlir.GoCreateFunctionType(b.ctx, receiver, inputs, results)
}

func (b *Builder) createStructType(ctx context.Context, T *types.Struct) mlir.Type {
	identifier := currentIdentifier(ctx)

	// Unset the name in a new context
	ctx = context.WithValue(ctx, identifierKey{}, "")

	var structType mlir.Type
	if len(identifier) > 0 {
		structType = mlir.GoCreateNamedStructType(b.ctx, identifier)

		// Prevent infinite recursion when mutually recursive types are encountered
		b.typeCache[T] = structType
	}

	// Create the struct field types
	var fieldNames []mlir.Attribute
	var fieldTags []mlir.Attribute
	var fieldTypes []mlir.Type
	for i := 0; i < T.NumFields(); i++ {
		fieldNames = append(fieldNames, mlir.StringAttrGet(b.ctx, T.Field(i).Name()))
		fieldTypes = append(fieldTypes, b.GetStoredType(ctx, T.Field(i).Type()))
		fieldTags = append(fieldTags, mlir.StringAttrGet(b.ctx, T.Tag(i)))
	}

	if len(identifier) > 0 {
		// Set the struct body
		mlir.GoSetStructTypeBody(structType, fieldNames, fieldTypes, fieldTags)
	} else {
		// Create a literal struct
		structType = mlir.GoCreateLiteralStructType(b.ctx, fieldNames, fieldTypes, fieldTags)
	}

	return structType
}

func (b *Builder) GetStoredType(ctx context.Context, T types.Type) mlir.Type {
	switch baseType(T).(type) {
	case *types.Signature:
		// This variable is a function, so use the _func struct type.
		return b._func
	default:
		return b.GetType(ctx, T)
	}
}

func (b *Builder) exprTypeHasFlags(ctx context.Context, expr ast.Expr, flags ...types.BasicInfo) bool {
	info := currentInfo(ctx)

	// Look the type information about the input expression.
	T := info.TypeOf(expr)
	return typeHasFlags(T, flags...)
}

func typeHasFlags(T types.Type, flags ...types.BasicInfo) bool {
	_T, ok := T.(*types.Basic)
	if !ok {
		_T, ok = T.Underlying().(*types.Basic)
	}

	if ok {
		for _, flag := range flags {
			if _T.Info()&flag == 0 {
				return false
			}
		}
		return true
	}
	return false
}

func isBasicKind(T types.Type, kind types.BasicKind) bool {
	if T, ok := T.(*types.Basic); ok {
		return T.Kind() == kind
	}
	return false
}

func isNil(T types.Type) bool {
	switch T := T.(type) {
	case *types.Tuple:
		return T == nil
	default:
		return isBasicKind(T, types.UntypedNil)
	}
}

func isUntyped(T types.Type) bool {
	if T == nil {
		return true
	}
	return typeHasFlags(T, types.IsUntyped)
}

func typeIs[KIND types.Type](T types.Type) bool {
	_, ok := T.(KIND)
	if !ok {
		_, ok = T.Underlying().(KIND)
	}
	return ok
}

func isPointer(T types.Type) bool {
	T = T.Underlying()
	switch T := T.(type) {
	case *types.Pointer:
		return true
	case *types.Basic:
		if T.Kind() == types.UnsafePointer {
			return true
		}
	}
	return false
}

func isGeneric(T types.Type) bool {
	if T, ok := T.(*types.Named); ok {
		return T.TypeParams().Len() > 0
	}
	return false
}

func isUnsafePointer(T types.Type) bool {
	if T, ok := T.Underlying().(*types.Basic); ok {
		return T.Kind() == types.UnsafePointer
	}
	return false
}

func findStructField(name string, T *types.Struct) (int, *types.Var) {
	for i := 0; i < T.NumFields(); i++ {
		v := T.Field(i)
		if name == v.Name() {
			return i, v
		}
	}
	return -1, nil
}

func baseStructTypeOf(T types.Type) *types.Struct {
	switch T := T.(type) {
	case *types.Named:
		return baseStructTypeOf(T.Underlying())
	case *types.Pointer:
		return baseStructTypeOf(T.Elem())
	case *types.Struct:
		return T
	default:
		return nil
	}
}

func baseType(T types.Type) types.Type {
	for {
		if named, ok := T.(*types.Named); ok {
			T = named.Underlying()
			continue
		}
		return T
	}
}

func (b *Builder) widthOf(T mlir.Type) int {
	switch {
	case mlir.GoTypeIsInteger(T):
		width := mlir.GoIntegerTypeGetWidth(T)
		if width == 0 {
			return int(b.config.Sizes.WordSize * 8)
		}
		return width
	case mlir.TypeIsAF32(T):
		return 32
	case mlir.TypeIsAF64(T):
		return 64
	default:
		panic("unhandled")
	}
}

func isSigned(T mlir.Type) bool {
	if !mlir.GoTypeIsInteger(T) {
		panic("invalid type")
	}
	return mlir.GoIntegerTypeIsSigned(T)
}

func isUnsigned(T mlir.Type) bool {
	if !mlir.GoTypeIsInteger(T) {
		panic("invalid type")
	}
	return mlir.GoIntegerTypeIsUnsigned(T)
}

func resolveType(ctx context.Context, T types.Type) types.Type {
	if isUntyped(T) {
		lhsTypes := currentLhsList(ctx)
		index := currentRhsIndex(ctx)
		if len(lhsTypes) > 0 {
			T = lhsTypes[index]
		} else {
			T = types.Default(T)
		}
	}
	return T
}
