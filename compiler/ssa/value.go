package ssa

import (
	"context"
	"go/types"

	"pkg.si-go.dev/go-mlir/mlir"
	"pkg.si-go.dev/sigo/goir/binding/goir"
)

type Value interface {
	Load(ctx context.Context, location mlir.LocationLike) mlir.Value
	Store(ctx context.Context, value mlir.ValueLike, location mlir.LocationLike)
	Pointer(ctx context.Context, location mlir.LocationLike) mlir.Value
	Type() mlir.Type
}

type ConstantValue struct {
	Emitter func(context.Context, mlir.LocationLike) mlir.Value
	T       mlir.TypeLike
	b       *Builder
}

func (c ConstantValue) Load(ctx context.Context, location mlir.LocationLike) mlir.Value {
	return c.Emitter(ctx, location)
}

func (c ConstantValue) Store(ctx context.Context, value mlir.ValueLike, location mlir.LocationLike) {
	panic("cannot store to constant")
}

func (c ConstantValue) Pointer(ctx context.Context, location mlir.LocationLike) mlir.Value {
	value := c.Emitter(ctx, location)
	return c.b.makeCopyOf(ctx, value, location)
}

func (c ConstantValue) Type() mlir.Type {
	return c.T.ToType()
}

type GlobalValue struct {
	symbol string
	T      mlir.TypeLike
	ctx    mlir.Context
}

func (g GlobalValue) Load(ctx context.Context, location mlir.LocationLike) mlir.Value {
	op := goir.NewLoadOperation(g.ctx, g.Pointer(ctx, location), g.Type(), location)
	appendOperation(ctx, op)
	return resultOf(op).AsValue()
}

func (g GlobalValue) Store(ctx context.Context, value mlir.ValueLike, location mlir.LocationLike) {
	op := goir.NewStoreOperation(g.ctx, value, g.Pointer(ctx, location), location)
	appendOperation(ctx, op)
}

func (g GlobalValue) Pointer(ctx context.Context, location mlir.LocationLike) mlir.Value {
	op := goir.NewAddressOfOperation(g.ctx, g.symbol, goir.NewPointerType(g.Type()), location)
	appendOperation(ctx, op)
	return resultOf(op).AsValue()
}

func (g GlobalValue) Type() mlir.Type {
	return g.T.ToType()
}

func (g GlobalValue) Initialize(ctx context.Context, builder *Builder, priority int, fn func(context.Context, *Builder) mlir.Value, location mlir.LocationLike) {
	// Find the operation for this global in the current module's symbol table.
	globalOp := builder.lookupSymbol(g.symbol)
	if globalOp.IsNull() {
		return
	}

	// Create the initializer region for this global.
	region := globalOp.Region(0)
	block := mlir.NewBlock(nil, nil)

	// Check if the operation already has a region.
	if blockHasTerminator(block) {
		// Cannot initialize the global more than once.
		return
	}

	// Set the initializer priority value attribute.
	priorityAttr := builder.int32Attr(int32(priority))
	globalOp.SetAttributeByName("go.ctor.priority", priorityAttr)

	// Create the initializer block.
	region.AppendOwnedBlock(block)

	newCtx := newGlobalContext(context.Background())
	if val := ctx.Value(jobQueueKey{}); val != nil {
		queue := val.(*jobQueue)
		newCtx = context.WithValue(newCtx, jobQueueKey{}, queue)
	}
	newCtx = newContextWithRegion(newCtx, region)
	newCtx = newContextWithCurrentBlock(newCtx)

	setCurrentBlock(newCtx, block)
	result := fn(newCtx, builder)

	// Create the terminator operation.
	yieldOp := goir.NewYieldOperation(builder.ctx, result, location)
	appendOperation(newCtx, yieldOp)
}

type LocalValue struct {
	ptr mlir.ValueLike
	T   mlir.TypeLike
	b   *Builder
	obj types.Object
}

func (l LocalValue) Load(ctx context.Context, location mlir.LocationLike) mlir.Value {
	op := goir.NewLoadOperation(l.b.ctx, l.ptr, l.Type(), location)
	appendOperation(ctx, op)
	return resultOf(op).AsValue()
}

func (l LocalValue) Store(ctx context.Context, value mlir.ValueLike, location mlir.LocationLike) {
	op := goir.NewStoreOperation(l.b.ctx, value, l.ptr, location)
	appendOperation(ctx, op)
}

func (l LocalValue) Pointer(ctx context.Context, location mlir.LocationLike) mlir.Value {
	return l.ptr.AsValue()
}

func (l LocalValue) Type() mlir.Type {
	return l.T.ToType()
}

type FreeVar struct {
	obj types.Object
	ptr mlir.Value // **void
	T   mlir.TypeLike
	b   *Builder
}

func (f FreeVar) Load(ctx context.Context, location mlir.LocationLike) mlir.Value {
	// Get the address of the value.
	addr := f.Pointer(ctx, location)

	// Load the actual value.
	loadOp := goir.NewLoadOperation(f.b.ctx, addr, f.T, location)
	appendOperation(ctx, loadOp)
	return resultOf(loadOp).AsValue()
}

func (f FreeVar) Store(ctx context.Context, value mlir.ValueLike, location mlir.LocationLike) {
	// Get the address of the value.
	addr := f.Pointer(ctx, location)

	// Store the value at the address.
	op := goir.NewStoreOperation(f.b.ctx, value, addr, location)
	appendOperation(ctx, op)
}

func (f FreeVar) Pointer(ctx context.Context, location mlir.LocationLike) mlir.Value {
	// Load the address of the value.
	loadOp := goir.NewLoadOperation(f.b.ctx, f.ptr, goir.NewPointerType(f.T), location)
	appendOperation(ctx, loadOp)
	return resultOf(loadOp).AsValue()
}

func (f FreeVar) Type() mlir.Type {
	return f.T.ToType()
}

type TempValue struct {
	ptr       mlir.ValueLike
	b         *Builder
	valueType mlir.TypeLike
}

func (t *TempValue) Load(ctx context.Context, location mlir.LocationLike) mlir.Value {
	op := goir.NewLoadOperation(t.b.ctx, t.ptr, t.Type(), location)
	appendOperation(ctx, op)
	return resultOf(op).AsValue()
}

func (t *TempValue) Store(ctx context.Context, value mlir.ValueLike, location mlir.LocationLike) {
	op := goir.NewStoreOperation(t.b.ctx, value, t.ptr, location)
	appendOperation(ctx, op)
}

func (t *TempValue) Pointer(ctx context.Context, location mlir.LocationLike) mlir.Value {
	return t.ptr.AsValue()
}

func (t *TempValue) Type() mlir.Type {
	if t.valueType == nil {
		ptrType, _ := goir.AsPointerType(t.ptr.Type())
		return ptrType.ElementType()
	}
	return t.valueType.ToType()
}

func (b *Builder) NewTempValue(ptr mlir.ValueLike) *TempValue {
	// TODO: Panic if the ptr value is not actually of the pointer type.
	return &TempValue{
		ptr: ptr,
		b:   b,
	}
}

func (b *Builder) NewTempValueWithType(ptr mlir.ValueLike, valueType mlir.TypeLike) *TempValue {
	return &TempValue{
		ptr:       ptr,
		b:         b,
		valueType: valueType,
	}
}
