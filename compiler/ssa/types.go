package ssa

import (
	"context"
	"fmt"
	"go/ast"
	"go/types"

	"pkg.si-go.dev/go-mlir/mlir"
	"pkg.si-go.dev/sigo/goir/binding/goir"
)

type typeCacheNestedLockKey struct{}

func (b *Builder) GetType(ctx context.Context, T types.Type) (result mlir.TypeLike) {
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
	case *types.Alias:
		return b.GetType(ctx, types.Unalias(T))
	case *types.Array:
		result = b.createArrayType(ctx, T)
	case *types.Basic:
		result = b.createBasicType(T)
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
		result = b.createSignatureType(ctx, T)
	case *types.Slice:
		result = b.createSliceType(ctx, T)
	case *types.Struct:
		result = b.createStructType(ctx, T)
	case *types.Tuple:
		println(T.String())
		panic("unreachable")
	case *types.TypeParam:
		// Look up the instantiated type in the data of the current function.
		typeMap := currentTypeMap(ctx)
		if typeMap == nil {
			panic("no type mapping exists in the current context")
		}

		concreteType := typeMap[T.Index()]
		if concreteType == T {
			panic("unreachable")
		} else if concreteType == nil {
			panic("no concrete type for type parameter could be determined")
		}

		return b.GetType(ctx, concreteType)
	default:
		panic("unhandled type: ")
	}

	if result == nil {
		panic("no type was created")
	}

	b.typeCache[T] = result
	return result
}

func (b *Builder) createArrayType(ctx context.Context, T *types.Array) goir.ArrayType {
	// Create the element type.
	elementType := b.GetStoredType(ctx, T.Elem())

	// Return the array type.
	return goir.NewArrayType(elementType, int(T.Len()))
}

func (b *Builder) createBasicType(T *types.Basic) mlir.TypeLike {
	switch T.Kind() {
	case types.Bool:
		return goir.NewBooleanType(b.ctx)
	case types.Int:
		return goir.NewSignedIntType(b.ctx, 0)
	case types.Uint:
		return goir.NewUnsignedIntType(b.ctx, 0)
	case types.Uintptr:
		return goir.NewUintptrType(b.ctx)
	case types.Int8:
		return goir.NewSignedIntType(b.ctx, 8)
	case types.Uint8:
		return goir.NewUnsignedIntType(b.ctx, 8)
	case types.Int16:
		return goir.NewSignedIntType(b.ctx, 16)
	case types.Uint16:
		return goir.NewUnsignedIntType(b.ctx, 16)
	case types.Int32:
		return goir.NewSignedIntType(b.ctx, 32)
	case types.Uint32:
		return goir.NewUnsignedIntType(b.ctx, 32)
	case types.Int64:
		return goir.NewSignedIntType(b.ctx, 64)
	case types.Uint64:
		return goir.NewUnsignedIntType(b.ctx, 64)
	case types.Float32:
		return mlir.NewFloatType(b.ctx, mlir.Float32)
	case types.Float64:
		return mlir.NewFloatType(b.ctx, mlir.Float64)
	case types.Complex64:
		return mlir.NewComplexType(mlir.NewFloatType(b.ctx, mlir.Float32))
	case types.Complex128:
		return mlir.NewComplexType(mlir.NewFloatType(b.ctx, mlir.Float64))
	case types.String:
		return goir.NewStringType(b.ctx)
	case types.UnsafePointer:
		return goir.NewUnsafePointerType(b.ctx)
	case types.UntypedBool:
		return goir.NewUntypedType(b.ctx, goir.BasicTypeBoolean)
	case types.UntypedComplex:
		return goir.NewUntypedType(b.ctx, goir.BasicTypeComplex)
	case types.UntypedFloat:
		return goir.NewUntypedType(b.ctx, goir.BasicTypeFloat)
	case types.UntypedInt:
		return goir.NewUntypedType(b.ctx, goir.BasicTypeInteger)
	case types.UntypedNil:
		return goir.NewUntypedType(b.ctx, goir.BasicTypeNil)
	case types.UntypedRune:
		return goir.NewUntypedType(b.ctx, goir.BasicTypeRune)
	case types.UntypedString:
		return goir.NewUntypedType(b.ctx, goir.BasicTypeString)
	default:
		panic(fmt.Sprintf("unknown basic type %+v", T.Kind()))
	}
}

func (b *Builder) createChanType(ctx context.Context, T *types.Chan) goir.ChanType {
	// Create the element type.
	elementType := b.GetStoredType(ctx, T.Elem())

	// Return the chan type
	switch T.Dir() {
	case types.SendRecv:
		return goir.NewChanType(elementType, goir.ChanDirectionSendRecv)
	case types.SendOnly:
		return goir.NewChanType(elementType, goir.ChanDirectionSendOnly)
	case types.RecvOnly:
		return goir.NewChanType(elementType, goir.ChanDirectionRecvOnly)
	default:
		panic("invalid chan direction")
	}
}

func (b *Builder) createInterfaceType(ctx context.Context, T *types.Interface) goir.InterfaceType {
	var methodNames []string
	var methods []goir.FunctionType

	// Get the identifier if this is actually a named interface type.
	identifier := currentIdentifier(ctx)

	// Unset the name in a new context.
	ctx = context.WithValue(ctx, identifierKey{}, "")

	// NOTE: Named interfaces need to declare the interface type first before creating the methods in order to prevent
	//       infinite recursion.
	if len(identifier) > 0 {
		// Create a named interface.
		result := goir.NewNamedInterfaceType(b.ctx, identifier)

		// Prevent infinite recursion when mutually recursive types are encountered.
		b.typeCache[T] = result

		// Create the function signatures.
		for i := 0; i < T.NumMethods(); i++ {
			method := T.Method(i)
			MT := b.createSignatureType(ctx, method.Type().(*types.Signature))
			methodNames = append(methodNames, method.Name())
			methods = append(methods, MT)
		}

		// Set the interface methods.
		goir.SetNamedInterfaceMethods(b.ctx, result, methodNames, methods)

		// Return the interface type.
		return result
	} else {
		// Create the function signatures
		for i := 0; i < T.NumMethods(); i++ {
			method := T.Method(i)
			MT := b.createSignatureType(ctx, method.Type().(*types.Signature))
			methodNames = append(methodNames, method.Name())
			methods = append(methods, MT)
		}

		// Create a literal interface
		return goir.NewInterfaceType(b.ctx, methodNames, methods)
	}
}

func (b *Builder) createNamedType(ctx context.Context, T *types.Named) goir.NamedType {
	// Format the qualified identifier for this type with respect to its origin package.
	identifier := qualifiedName(T.Obj().Name(), T.Obj().Pkg())

	// Add the identifier to the current context.
	ctx = newContextWithIdentifier(ctx, identifier)

	// Create the underlying type.
	underlyingType := b.GetType(ctx, T.Underlying())
	underlyingTypeHash := goir.TypeHash(underlyingType)

	if T.TypeArgs().Len() > 0 {
		identifier += fmt.Sprintf("$%X", underlyingTypeHash)
	}

	// Collect method symbols.
	entries := make([]mlir.AttributeLike, T.NumMethods())
	for i := 0; i < T.NumMethods(); i++ {
		method := T.Method(i)
		symbol := qualifiedFuncName(method)
		refAttr := mlir.NewFlatSymbolRefAttr(b.ctx, symbol)
		entries[i] = refAttr
	}
	methodSymbols := mlir.NewArrayAttr(b.ctx, entries)

	// Create the named type now.
	result := goir.NewNamedType(underlyingType, identifier, methodSymbols)

	// Prevent infinite recursion within metadata and mutually recursive types by mapping the named type now.
	b.typeCache[T] = result

	return result
}

func (b *Builder) createMapType(ctx context.Context, T *types.Map) goir.MapType {
	// Create the key type.
	keyType := b.GetStoredType(ctx, T.Key())

	// Create the element type.
	elementType := b.GetStoredType(ctx, T.Elem())

	// Return the map type.
	return goir.NewMapType(keyType, elementType)
}

func (b *Builder) createPointerType(ctx context.Context, T *types.Pointer) goir.PointerType {
	elementType := b.GetStoredType(ctx, T.Elem())
	return goir.NewPointerType(elementType)
}

func (b *Builder) pointerOf(ctx context.Context, T types.Type) goir.PointerType {
	ptrType := types.NewPointer(T)
	return b.GetStoredType(ctx, ptrType).(goir.PointerType)
}

func (b *Builder) funcPointerOf(ctx context.Context, T *types.Signature) goir.PointerType {
	fnT := b.GetType(ctx, T)
	return goir.NewPointerType(fnT)
}

func (b *Builder) createSliceType(ctx context.Context, T *types.Slice) goir.SliceType {
	// Create the element type
	elementType := b.GetStoredType(ctx, T.Elem())

	// Create the slice type
	return goir.NewSliceType(elementType)
}

func (b *Builder) createSignatureType(ctx context.Context, T *types.Signature) goir.FunctionType {
	var receiver mlir.TypeLike
	var inputs []mlir.TypeLike
	var results []mlir.TypeLike

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

	return goir.NewFunctionType(b.ctx, receiver, inputs, results)
}

func (b *Builder) createStructType(ctx context.Context, T *types.Struct) goir.StructType {
	identifier := currentIdentifier(ctx)

	// Unset the name in a new context
	ctx = context.WithValue(ctx, identifierKey{}, "")

	var structType goir.StructType
	if len(identifier) > 0 {
		structType = goir.NewNamedStructType(b.ctx, identifier)

		// Prevent infinite recursion when mutually recursive types are encountered
		b.typeCache[T] = structType
	}

	// Create the struct field types
	var fieldNames []mlir.StringAttr
	var fieldTags []mlir.StringAttr
	var fieldTypes []mlir.TypeLike
	for i := 0; i < T.NumFields(); i++ {
		fieldNames = append(fieldNames, mlir.NewStringAttr(b.ctx, T.Field(i).Name()))
		fieldTypes = append(fieldTypes, b.GetStoredType(ctx, T.Field(i).Type()))
		fieldTags = append(fieldTags, mlir.NewStringAttr(b.ctx, T.Tag(i)))
	}

	if len(identifier) > 0 {
		// Set the struct body
		goir.SetStructTypeBody(structType, fieldNames, fieldTypes, fieldTags)
	} else {
		// Create a literal struct
		structType = goir.NewLiteralStructType(b.ctx, fieldNames, fieldTypes, fieldTags)
	}

	return structType
}

func (b *Builder) GetStoredType(ctx context.Context, T types.Type) mlir.TypeLike {
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
	T = types.Unalias(T)
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
	T = types.Unalias(T)
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
	T = types.Unalias(T)
	for {
		if named, ok := T.(*types.Named); ok {
			T = named.Underlying()
			continue
		}
		return T
	}
}

func (b *Builder) widthOf(T mlir.TypeLike) int {
	if complexType, ok := mlir.AsComplexType(T); ok {
		return b.widthOf(complexType.ElementType()) * 2
	} else if floatType, ok := mlir.AsFloatType(T); ok {
		return floatType.Width()
	} else if intType, ok := goir.AsIntegerType(T); ok {
		width := intType.Width()
		if width == 0 {
			return int(b.config.Sizes.WordSize * 8)
		}
		return width
	} else if untyped, ok := goir.AsUntypedType(T); ok {
		switch untyped.BasicKind() {
		case goir.BasicTypeComplex:
			return 64
		case goir.BasicTypeFloat:
			return 64
		case goir.BasicTypeInteger:
			return int(b.config.Sizes.WordSize) * 8
		default:
			panic("invalid basic type")
		}
	}
	panic("unreachable")
}

func isSigned(T mlir.TypeLike) bool {
	if goir.TypeIsUntyped(T) {
		return true
	}

	intT, ok := goir.AsIntegerType(T)
	if !ok {
		panic("invalid type")
	}
	return intT.IsSigned()
}

func isUnsigned(T mlir.TypeLike) bool {
	if goir.TypeIsUntyped(T) {
		return true
	}

	intT, ok := goir.AsIntegerType(T)
	if !ok {
		panic("invalid type")
	}
	return intT.IsUnsigned()
}

func isUintptr(T mlir.TypeLike) bool {
	if goir.TypeIsUntyped(T) {
		return true
	}

	intT, ok := goir.AsIntegerType(T)
	if !ok {
		panic("invalid type")
	}
	return intT.IsUintptr()
}

func resolveType(ctx context.Context, T types.Type) types.Type {
	if typeParam, ok := T.(*types.TypeParam); ok {
		typeMap := currentTypeMap(ctx)
		if typeMap != nil {
			if resolvedT, ok := typeMap[typeParam.Index()]; ok {
				return resolvedT
			}
		}
	}
	return T
}
