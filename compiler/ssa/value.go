package ssa

import (
	"context"
	"go/types"
	"omibyte.io/sigo/mlir"
)

type Value interface {
	Load(ctx context.Context, location mlir.Location) mlir.Value
	Store(ctx context.Context, value mlir.Value, location mlir.Location)
	Pointer(ctx context.Context, location mlir.Location) mlir.Value
	Type() mlir.Type
}

type ConstantValue struct {
	Emitter func(context.Context, mlir.Location) mlir.Value
	T       mlir.Type
	b       *Builder
}

func (c ConstantValue) Load(ctx context.Context, location mlir.Location) mlir.Value {
	return c.Emitter(ctx, location)
}

func (c ConstantValue) Store(ctx context.Context, value mlir.Value, location mlir.Location) {
	panic("cannot store to constant")
}

func (c ConstantValue) Pointer(ctx context.Context, location mlir.Location) mlir.Value {
	value := c.Emitter(ctx, location)
	return c.b.makeCopyOf(ctx, value, location)
}

func (c ConstantValue) Type() mlir.Type {
	return c.T
}

type GlobalValue struct {
	symbol string
	T      mlir.Type
	ctx    mlir.Context
}

func (g GlobalValue) Load(ctx context.Context, location mlir.Location) mlir.Value {
	op := mlir.GoCreateLoadOperation(g.ctx, g.Pointer(ctx, location), g.Type(), location)
	appendOperation(ctx, op)
	return resultOf(op)
}

func (g GlobalValue) Store(ctx context.Context, value mlir.Value, location mlir.Location) {
	op := mlir.GoCreateStoreOperation(g.ctx, value, g.Pointer(ctx, location), location)
	appendOperation(ctx, op)
}

func (g GlobalValue) Pointer(ctx context.Context, location mlir.Location) mlir.Value {
	op := mlir.GoCreateAddressOfOperation(g.ctx, g.symbol, mlir.GoCreatePointerType(g.Type()), location)
	appendOperation(ctx, op)
	return resultOf(op)
}

func (g GlobalValue) Type() mlir.Type {
	return g.T
}

func (g GlobalValue) Initialize(ctx context.Context, builder *Builder, priority int, fn func(context.Context, *Builder) mlir.Value, location mlir.Location) {
	// Find the operation for this global in the current module's symbol table.
	globalOp := builder.lookupSymbol(g.symbol)
	if globalOp == nil {
		return
	}

	// Create the initializer region for this global.
	region := mlir.OperationGetFirstRegion(globalOp)
	block := mlir.BlockCreate2(nil, nil)

	// Check if the operation already has a region.
	if !mlir.OperationIsNull(mlir.BlockGetTerminator(block)) {
		// Cannot initialize the global more than once.
		return
	}

	// Set the initializer priority value attribute.
	priorityAttr := builder.int32Attr(int32(priority))
	mlir.OperationSetAttributeByName(globalOp, "go.ctor.priority", priorityAttr)

	// Create the initializer block.
	mlir.RegionAppendOwnedBlock(region, block)

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
	yieldOp := mlir.GoCreateYieldOperation(builder.ctx, result, location)
	appendOperation(newCtx, yieldOp)
}

type LocalValue struct {
	ptr mlir.Value
	T   mlir.Type
	b   *Builder
}

func (l LocalValue) Load(ctx context.Context, location mlir.Location) mlir.Value {
	op := mlir.GoCreateLoadOperation(l.b.ctx, l.ptr, l.Type(), location)
	appendOperation(ctx, op)
	return resultOf(op)
}

func (l LocalValue) Store(ctx context.Context, value mlir.Value, location mlir.Location) {
	op := mlir.GoCreateStoreOperation(l.b.ctx, value, l.ptr, location)
	appendOperation(ctx, op)
}

func (l LocalValue) Pointer(ctx context.Context, location mlir.Location) mlir.Value {
	return l.ptr
}

func (l LocalValue) Type() mlir.Type {
	return l.T
}

type FreeVar struct {
	obj types.Object
	ptr mlir.Value // **void
	T   mlir.Type
	b   *Builder
}

func (f FreeVar) Load(ctx context.Context, location mlir.Location) mlir.Value {
	// Get the address of the value.
	addr := f.Pointer(ctx, location)

	// Load the actual value.
	loadOp := mlir.GoCreateLoadOperation(f.b.ctx, addr, f.T, location)
	appendOperation(ctx, loadOp)
	return resultOf(loadOp)
}

func (f FreeVar) Store(ctx context.Context, value mlir.Value, location mlir.Location) {
	// Get the address of the value.
	addr := f.Pointer(ctx, location)

	// Store the value at the address.
	op := mlir.GoCreateStoreOperation(f.b.ctx, value, addr, location)
	appendOperation(ctx, op)
}

func (f FreeVar) Pointer(ctx context.Context, location mlir.Location) mlir.Value {
	// Load the address of the value.
	loadOp := mlir.GoCreateLoadOperation(f.b.ctx, f.ptr, mlir.GoCreatePointerType(f.T), location)
	appendOperation(ctx, loadOp)
	return resultOf(loadOp)
}

func (f FreeVar) Type() mlir.Type {
	return f.T
}

type TempValue struct {
	ptr mlir.Value
	b   *Builder
}

func (t *TempValue) Load(ctx context.Context, location mlir.Location) mlir.Value {
	op := mlir.GoCreateLoadOperation(t.b.ctx, t.ptr, t.Type(), location)
	appendOperation(ctx, op)
	return resultOf(op)
}

func (t *TempValue) Store(ctx context.Context, value mlir.Value, location mlir.Location) {
	op := mlir.GoCreateStoreOperation(t.b.ctx, value, t.ptr, location)
	appendOperation(ctx, op)
}

func (t *TempValue) Pointer(ctx context.Context, location mlir.Location) mlir.Value {
	return t.ptr
}

func (t *TempValue) Type() mlir.Type {
	return mlir.GoPointerTypeGetElementType(mlir.ValueGetType(t.ptr))
}

func (b *Builder) NewTempValue(ptr mlir.Value) *TempValue {
	// TODO: Panic if the ptr value is not actually of the pointer type.
	return &TempValue{
		ptr: ptr,
		b:   b,
	}
}
