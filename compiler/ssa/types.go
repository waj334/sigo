package ssa

import (
	"context"
	"fmt"
	"go/ast"
	"go/types"

	"pkg.si-go.dev/go-mlir/mlir"
	"pkg.si-go.dev/sigo/goir/binding/goir"
)

// GetType is the public entry point for resolving a Go type to an MLIR type.
// It acquires the type cache lock and delegates to getTypeImpl.
func (b *Builder) GetType(ctx context.Context, T types.Type) mlir.TypeLike {
	b.typeCacheMutex.Lock()
	defer b.typeCacheMutex.Unlock()
	return b.getTypeImpl(ctx, T)
}

// getTypeImpl does the actual type resolution without locking. All internal
// recursive calls (from create* methods) must use this instead of GetType.
func (b *Builder) getTypeImpl(ctx context.Context, T types.Type) (result mlir.TypeLike) {
	// First attempt to resolve type params.
	T = resolveType(ctx, T)

	// For TypeParam-containing types within a generic instance, use a
	// per-instance cache to avoid cross-instance pollution while still
	// breaking recursion for self-referential types within this instance.
	instCache := currentInstanceTypeCache(ctx)
	useInstCache := instCache != nil && containsTypeParam(T)

	if useInstCache {
		if cached, ok := instCache[T]; ok {
			return cached
		}
		// Check if this type is already being created in the current call
		// chain. This detects recursion caused by TypeParam index collisions
		// across different generic scopes (e.g., TypeParam from type A at
		// index 0 is incorrectly resolved using type B's typeMap which also
		// maps index 0). Fall back to the global cache to break the cycle.
		if procSet := currentTypeProcessingSet(ctx); procSet != nil {
			if procSet[T] {
				useInstCache = false
			} else {
				procSet[T] = true
				defer delete(procSet, T)
			}
		}
	}
	if !useInstCache {
		if cached, ok := b.typeCache[T]; ok {
			return cached
		}
	}

	switch T := T.(type) {
	case *types.Alias:
		return b.getTypeImpl(ctx, types.Unalias(T))
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

		// Detect self-referential type parameter resolution (e.g., TypeParam
		// resolves to *TypeParam due to index collision across generic scopes).
		// Replace the TypeParam in the resolved type with the concrete mapping
		// to break the cycle.
		concreteType = resolveTypeInTypeMap(concreteType, typeMap)

		return b.getTypeImpl(ctx, concreteType)
	default:
		panic(fmt.Sprintf("unhandled type: %T %v", T, T))
	}

	if result == nil {
		panic("no type was created")
	}

	if useInstCache {
		instCache[T] = result
	} else {
		b.typeCache[T] = result
	}
	return result
}

func (b *Builder) createArrayType(ctx context.Context, T *types.Array) goir.ArrayType {
	// Create the element type.
	elementType := b.getStoredTypeImpl(ctx, T.Elem())

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
	elementType := b.getStoredTypeImpl(ctx, T.Elem())

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

	// For literal interfaces with methods, use a synthetic name so we can
	// break self-referencing cycles (e.g., interface{ Timeout() bool } where
	// the method receiver is the interface itself).
	if len(identifier) == 0 && T.NumMethods() > 0 {
		identifier = T.String()
	}

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
	identifier := qualifiedName(T.Obj().Name(), T.Obj().Pkg())
	ctx = newContextWithIdentifier(ctx, identifier)
	underlyingType := b.getTypeImpl(ctx, T.Underlying())
	underlyingTypeHash := goir.TypeHash(underlyingType)
	if T.TypeArgs().Len() > 0 {
		identifier += fmt.Sprintf("$%X", underlyingTypeHash)
	}

	// Direct methods.
	entries := make([]mlir.AttributeLike, 0, T.NumMethods())
	for i := 0; i < T.NumMethods(); i++ {
		method := T.Method(i)
		entries = append(entries,
			mlir.NewFlatSymbolRefAttr(b.ctx, qualifiedFuncName(method)))
	}

	// Promoted methods (struct underlying only — interfaces don't promote,
	// and other underlying kinds can't have embedded fields).
	if _, isStruct := types.Unalias(T.Underlying()).(*types.Struct); isStruct {
		mset := types.NewMethodSet(types.NewPointer(T))
		for i := 0; i < mset.Len(); i++ {
			sel := mset.At(i)
			if len(sel.Index()) == 1 {
				// Directly declared — already in the entries above.
				continue
			}
			trampSym := promotedTrampolineSymbol(T, sel.Obj().Name())
			entries = append(entries,
				mlir.NewFlatSymbolRefAttr(b.ctx, trampSym))
		}
	}

	methodSymbols := mlir.NewArrayAttr(b.ctx, entries)
	result := goir.NewNamedType(underlyingType, identifier, methodSymbols)

	// Existing cache/recursion handling unchanged.
	if instCache := currentInstanceTypeCache(ctx); instCache != nil && containsTypeParam(T) {
		procSet := currentTypeProcessingSet(ctx)
		if procSet != nil && procSet[T] {
			b.typeCache[T] = result
		} else {
			instCache[T] = result
		}
	} else {
		b.typeCache[T] = result
	}
	return result
}

func (b *Builder) createMapType(ctx context.Context, T *types.Map) goir.MapType {
	// Create the key type.
	keyType := b.getStoredTypeImpl(ctx, T.Key())

	// Create the element type.
	elementType := b.getStoredTypeImpl(ctx, T.Elem())

	// Return the map type.
	return goir.NewMapType(keyType, elementType)
}

func (b *Builder) createPointerType(ctx context.Context, T *types.Pointer) goir.PointerType {
	// Resolve TypeParams in the element type so that pointers to generic type
	// parameters (e.g., *T where T = SomeNamedType) go through the deferred
	// pointer path and produce the same MLIR type as direct *SomeNamedType.
	elem := resolveType(ctx, T.Elem())

	// Guard against TypeParam index collision across generic scopes: if
	// resolving the element produces this pointer type itself, the TypeParam
	// belongs to a different generic scope and was incorrectly resolved.
	// Fall back to the global cache to break the cycle.
	if elem == types.Type(T) {
		if cached, ok := b.typeCache[T]; ok {
			return cached.(goir.PointerType)
		}
	}

	// Pointers to function signatures are stored as pointers to the runtime._func
	// struct.  Use the pre-built _funcPtr (a deferred pointer keyed by "runtime._func")
	// so that they share MLIR type identity with direct *runtime._func pointers.
	if _, ok := elem.(*types.Signature); ok {
		return b._funcPtr.(goir.PointerType)
	}

	if namedType, ok := resolveToNamed(elem); ok {
		// Determine the fully qualified name of the element type.
		qualName := qualifiedName(namedType.Obj().Name(), namedType.Obj().Pkg())

		// Create the pointer type now. It's element type will be set later.
		ptrType := goir.NewDeferredPointerType(b.ctx, qualName)

		// It's possible for this pointer type to already have been created. Check that the element type is null before
		// attempting to set it.
		// Cache this pointer type now to break any recursion.
		// When in recursion-fallback mode (processing set has T), write to
		// the global cache so the cycle can be broken on re-entry.
		if instCache := currentInstanceTypeCache(ctx); instCache != nil && containsTypeParam(T) {
			procSet := currentTypeProcessingSet(ctx)
			if procSet != nil && procSet[T] {
				b.typeCache[T] = ptrType
			} else {
				instCache[T] = ptrType
			}
		} else {
			b.typeCache[T] = ptrType
		}

		// Create the element type.
		elementType := b.getStoredTypeImpl(ctx, namedType)

		// Finally, set the pointer type's element type.
		if ptrType.ElementType().IsNull() {
			ptrType.SetElementType(elementType)
		}

		return ptrType
	}

	elementType := b.getStoredTypeImpl(ctx, T.Elem())
	return goir.NewPointerType(elementType)
}

// resolveToNamed peels through aliases to find a *types.Named.
func resolveToNamed(T types.Type) (*types.Named, bool) {
	T = types.Unalias(T)
	named, ok := T.(*types.Named)
	return named, ok
}

func (b *Builder) pointerOf(ctx context.Context, T types.Type) goir.PointerType {
	return b.GetType(ctx, types.NewPointer(T)).(goir.PointerType)
}

func (b *Builder) funcPointerOf(ctx context.Context, T *types.Signature) goir.PointerType {
	// Create a pointer to the actual MLIR function type (not the _func struct).
	// This bypasses createPointerType which would redirect Signature pointers
	// to the _funcPtr deferred pointer.
	b.typeCacheMutex.Lock()
	defer b.typeCacheMutex.Unlock()
	fnType := b.getTypeImpl(ctx, T)
	return goir.NewPointerType(fnType)
}

func (b *Builder) createSliceType(ctx context.Context, T *types.Slice) goir.SliceType {
	// Create the element type
	elementType := b.getStoredTypeImpl(ctx, T.Elem())

	// Create the slice type
	return goir.NewSliceType(elementType)
}

func (b *Builder) createSignatureType(ctx context.Context, T *types.Signature) goir.FunctionType {
	var receiver mlir.TypeLike
	var inputs []mlir.TypeLike
	var results []mlir.TypeLike

	if T.Recv() != nil {
		// The receiver is always the first parameter to a method.
		receiver = b.getStoredTypeImpl(ctx, T.Recv().Type())
	}

	for i := 0; i < T.Params().Len(); i++ {
		inputs = append(inputs, b.getStoredTypeImpl(ctx, T.Params().At(i).Type()))
	}

	for i := 0; i < T.Results().Len(); i++ {
		results = append(results, b.getStoredTypeImpl(ctx, T.Results().At(i).Type()))
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
		fieldTypes = append(fieldTypes, b.getStoredTypeImpl(ctx, T.Field(i).Type()))
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
	b.typeCacheMutex.Lock()
	defer b.typeCacheMutex.Unlock()
	return b.getStoredTypeImpl(ctx, T)
}

// getStoredTypeImpl is the lock-free version of GetStoredType for internal use.
func (b *Builder) getStoredTypeImpl(ctx context.Context, T types.Type) mlir.TypeLike {
	// First attempt to resolve type params.
	T = resolveType(ctx, T)

	switch baseType(T).(type) {
	case *types.Signature:
		// This variable is a function, so use the _func struct type.
		return b._func
	default:
		return b.getTypeImpl(ctx, T)
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

// containsTypeParam reports whether T directly or transitively contains a
// types.TypeParam.  It does NOT follow Named.Underlying() to avoid false
// positives on concrete named types that happen to be defined in a generic
// context; Named types are only flagged when they carry unsubstituted type
// arguments.
func containsTypeParam(T types.Type) bool {
	switch T := T.(type) {
	case *types.TypeParam:
		return true
	case *types.Pointer:
		return containsTypeParam(T.Elem())
	case *types.Array:
		return containsTypeParam(T.Elem())
	case *types.Slice:
		return containsTypeParam(T.Elem())
	case *types.Map:
		return containsTypeParam(T.Key()) || containsTypeParam(T.Elem())
	case *types.Chan:
		return containsTypeParam(T.Elem())
	case *types.Named:
		if ta := T.TypeArgs(); ta != nil {
			for i := range ta.Len() {
				if containsTypeParam(ta.At(i)) {
					return true
				}
			}
		}
		return false
	case *types.Signature:
		for i := range T.Params().Len() {
			if containsTypeParam(T.Params().At(i).Type()) {
				return true
			}
		}
		for i := range T.Results().Len() {
			if containsTypeParam(T.Results().At(i).Type()) {
				return true
			}
		}
		if T.Recv() != nil {
			return containsTypeParam(T.Recv().Type())
		}
		return false
	default:
		return false
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

// resolveTypeInTypeMap resolves any TypeParams embedded in T using the given
// typeMap, with cycle detection.  If a TypeParam resolves to a type that
// transitively contains the same TypeParam index, the cycle is broken by
// leaving the inner TypeParam unresolved (it will be resolved by getTypeImpl's
// normal TypeParam handling on the next recursion).
func resolveTypeInTypeMap(T types.Type, typeMap TypeParamMap) types.Type {
	return resolveTypeInTypeMapImpl(T, typeMap, map[int]bool{})
}

func resolveTypeInTypeMapImpl(T types.Type, typeMap TypeParamMap, visited map[int]bool) types.Type {
	switch T := T.(type) {
	case *types.TypeParam:
		if visited[T.Index()] {
			// Cycle detected — leave this TypeParam unresolved to break it.
			return T
		}
		if resolved := typeMap[T.Index()]; resolved != nil && resolved != T {
			visited[T.Index()] = true
			result := resolveTypeInTypeMapImpl(resolved, typeMap, visited)
			delete(visited, T.Index())
			return result
		}
		return T
	case *types.Pointer:
		elem := resolveTypeInTypeMapImpl(T.Elem(), typeMap, visited)
		if elem == T.Elem() {
			return T
		}
		return types.NewPointer(elem)
	case *types.Slice:
		elem := resolveTypeInTypeMapImpl(T.Elem(), typeMap, visited)
		if elem == T.Elem() {
			return T
		}
		return types.NewSlice(elem)
	case *types.Array:
		elem := resolveTypeInTypeMapImpl(T.Elem(), typeMap, visited)
		if elem == T.Elem() {
			return T
		}
		return types.NewArray(elem, T.Len())
	case *types.Map:
		key := resolveTypeInTypeMapImpl(T.Key(), typeMap, visited)
		val := resolveTypeInTypeMapImpl(T.Elem(), typeMap, visited)
		if key == T.Key() && val == T.Elem() {
			return T
		}
		return types.NewMap(key, val)
	case *types.Chan:
		elem := resolveTypeInTypeMapImpl(T.Elem(), typeMap, visited)
		if elem == T.Elem() {
			return T
		}
		return types.NewChan(T.Dir(), elem)
	case *types.Named:
		typeArgs := T.TypeArgs()
		if typeArgs == nil || typeArgs.Len() == 0 {
			return T
		}
		newArgs := make([]types.Type, typeArgs.Len())
		changed := false
		for i := 0; i < typeArgs.Len(); i++ {
			newArgs[i] = resolveTypeInTypeMapImpl(typeArgs.At(i), typeMap, visited)
			if newArgs[i] != typeArgs.At(i) {
				changed = true
			}
		}
		if !changed {
			return T
		}
		inst, err := types.Instantiate(nil, T.Origin(), newArgs, false)
		if err != nil {
			panic(fmt.Sprintf("failed to instantiate type in resolveTypeInTypeMap: %v", err))
		}
		return inst
	default:
		return T
	}
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
