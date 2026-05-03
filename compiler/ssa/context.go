package ssa

import (
	"context"
	"go/types"

	"pkg.si-go.dev/go-mlir/mlir"
	"pkg.si-go.dev/sigo/goir/binding/goir"
)

type (
	regionKey            struct{}
	blockKey             struct{}
	successorBlockKey    struct{}
	predecessorBlockKey  struct{}
	fallthroughBlockKey  struct{}
	labeledBlocksKey     struct{}
	identifierKey        struct{}
	funcDataKey          struct{}
	globalKey            struct{}
	jobQueueKey          struct{}
	infoKey              struct{}
	typeMapKey           struct{}
	instanceTypeCacheKey struct{}
	typeProcessingSetKey struct{}
	scopeKey             struct{}
	rangeFuncFrameKey    struct{}
	pendingRangeLabelKey struct{}
)

// newContextWithPendingRangeLabel attaches a label name that the next
// emitted range-over-func statement should record on its frame, so that
// labeled break/continue statements inside the body can target it. Cleared
// by emitFuncRange after consumption.
func newContextWithPendingRangeLabel(ctx context.Context, label string) context.Context {
	return context.WithValue(ctx, pendingRangeLabelKey{}, label)
}

func currentPendingRangeLabel(ctx context.Context) string {
	if val := ctx.Value(pendingRangeLabelKey{}); val != nil {
		return val.(string)
	}
	return ""
}

// rangeFuncFrame is installed in the context of a range-over-func yield
// closure's body. It tells emitBranchStatement and emitReturn how to leave
// the closure correctly so the enclosing range statement behaves like the
// user wrote a normal for-loop body.
//
//   - `continue`     → branch to fallthroughBlock, which returns true.
//   - `break`        → emit `return false` from the closure. The iterator
//     observes false and stops; control falls through.
//   - `return X1...` → store result values into the captured resultTempVars
//     slots, store returnSentinel to the captured stateVar
//     slot, then emit `return false`. After the iter call,
//     emitFuncRange tests the state and either issues the
//     enclosing function's Return op or falls through.
type rangeFuncFrame struct {
	// parent is the enclosing range-over-func frame, if any. nil means this
	// is the OUTERMOST rangefunc within the current function — its
	// stateVar / resultTempVars / outerSig live in the current function.
	// Non-nil means this is a NESTED rangefunc; stateVar / resultTempVars /
	// outerSig point at the OUTERMOST frame's slots (so a `return` from
	// any depth writes to the same place, and the outermost post-iter
	// dispatch loads from there).
	parent *rangeFuncFrame

	// fallthroughBlock is the closure-local block that emits `return true`.
	// Unlabeled `continue` branches here. The block is appended to the
	// closure's region by emitFuncRange's bodyContextHook.
	fallthroughBlock mlir.Block

	// stateVar is a hidden int slot in the OUTERMOST enclosing function.
	// Default zero means "not returning"; returnSentinel means a return was
	// taken inside the body. nil disables Phase 2B handling (Phase 2A
	// subset).
	stateVar types.Object

	// returnSentinel is the value stored to *stateVar to signal that a body
	// `return` occurred.
	returnSentinel int64

	// resultTempVars are hidden temp slots in the OUTERMOST enclosing
	// function, one per result of the outermost function's signature.
	// emitReturn stores computed result values here before setting state
	// and emitting return false. The outermost frame's post-iter loads
	// them and emits the actual outer Return. May be empty for
	// void-returning outermost functions.
	resultTempVars []types.Object

	// outerSig is the OUTERMOST enclosing function's signature. emitReturn
	// under the override uses this to type the result values from the
	// user's return statement (the closure's own signature is
	// `func(...) bool` and would give the wrong number of result types).
	outerSig *types.Signature

	// label is the source-level label of the for-statement creating this
	// frame, or "" if the for-statement is unlabeled. Used by
	// emitBranchStatement to resolve `break L` / `continue L` to a frame.
	label string

	// depth is the nesting level of this frame: 0 for the outermost
	// rangefunc within the enclosing function, 1 for one nested level, etc.
	// Used in the state encoding for labeled break/continue: the sentinel
	// value identifies which frame depth the action targets.
	depth int

	// stoppedVar is a hidden bool slot per frame. Set to true before this
	// frame's closure returns false to its iterator. Checked at the start
	// of each closure invocation; if already true, the closure panics
	// (yield called after the iterator was told to stop). Implements the
	// Go 1.23 yield-after-stop runtime check.
	stoppedVar types.Object
}

// findRangeFuncFrameByLabel walks the frame stack (current → parent)
// looking for the frame whose source-level label matches `label`. Returns
// nil if no such frame is in scope (caller should fall back to the
// existing labeled-block mechanism for non-rangefunc loop labels).
func findRangeFuncFrameByLabel(start *rangeFuncFrame, label string) *rangeFuncFrame {
	for f := start; f != nil; f = f.parent {
		if f.label == label {
			return f
		}
	}
	return nil
}

func newContextWithRangeFuncFrame(ctx context.Context, frame *rangeFuncFrame) context.Context {
	return context.WithValue(ctx, rangeFuncFrameKey{}, frame)
}

func currentRangeFuncFrame(ctx context.Context) *rangeFuncFrame {
	if val := ctx.Value(rangeFuncFrameKey{}); val != nil {
		return val.(*rangeFuncFrame)
	}
	return nil
}

type blockWithArgs struct {
	block mlir.Block
	args  []mlir.ValueLike
}

func newGlobalContext(ctx context.Context) context.Context {
	return context.WithValue(ctx, globalKey{}, true)
}

func isGlobalContext(ctx context.Context) bool {
	if val := ctx.Value(globalKey{}); val != nil {
		return val.(bool)
	}
	return false
}

func newContextWithRegion(ctx context.Context, region mlir.Region) context.Context {
	return context.WithValue(ctx, regionKey{}, region)
}

func currentRegion(ctx context.Context) mlir.Region {
	if val := ctx.Value(regionKey{}); val != nil {
		return val.(mlir.Region)
	}
	return mlir.Region{}
}

func newContextWithIdentifier(ctx context.Context, identifier string) context.Context {
	return context.WithValue(ctx, identifierKey{}, identifier)
}

func currentIdentifier(ctx context.Context) string {
	if val := ctx.Value(identifierKey{}); val != nil {
		return val.(string)
	}
	return ""
}

func newContextWithFuncData(ctx context.Context, data *funcData) context.Context {
	return context.WithValue(ctx, funcDataKey{}, data)
}

func currentFuncData(ctx context.Context) *funcData {
	if val := ctx.Value(funcDataKey{}); val != nil {
		return val.(*funcData)
	}
	return nil
}

func newContextWithCurrentBlock(ctx context.Context) context.Context {
	var block *mlir.Block
	ctx = context.WithValue(ctx, blockKey{}, &block)
	return ctx
}

func currentBlock(ctx context.Context) mlir.Block {
	if val := ctx.Value(blockKey{}); val != nil {
		blockPtr := *(val.(**mlir.Block))
		if blockPtr != nil {
			return *blockPtr
		}
	}
	return mlir.Block{}
}

func setCurrentBlock(ctx context.Context, target mlir.Block) {
	if val := ctx.Value(blockKey{}); val != nil {
		blockPtr := val.(**mlir.Block)
		*blockPtr = &target
		return
	}
	panic("No block pointer in context")
}

func newContextWithSuccessorBlock(ctx context.Context, block mlir.Block, args []mlir.ValueLike) context.Context {
	return context.WithValue(ctx, successorBlockKey{}, blockWithArgs{block, args})
}

func currentSuccessorBlock(ctx context.Context) (mlir.Block, []mlir.ValueLike) {
	if val := ctx.Value(successorBlockKey{}); val != nil {
		b := val.(blockWithArgs)
		return b.block, b.args
	}
	return mlir.Block{}, nil
}

func newContextWithPredecessorBlock(ctx context.Context, block mlir.Block, args []mlir.ValueLike) context.Context {
	return context.WithValue(ctx, predecessorBlockKey{}, blockWithArgs{block, args})
}

func currentPredecessorBlock(ctx context.Context) (mlir.Block, []mlir.ValueLike) {
	if val := ctx.Value(predecessorBlockKey{}); val != nil {
		b := val.(blockWithArgs)
		return b.block, b.args
	}
	return mlir.Block{}, nil
}

func newContextWithFallthroughBlock(ctx context.Context, block mlir.Block, args []mlir.ValueLike) context.Context {
	return context.WithValue(ctx, fallthroughBlockKey{}, blockWithArgs{block, args})
}

func currentFallthroughBlock(ctx context.Context) (mlir.Block, []mlir.ValueLike) {
	if val := ctx.Value(fallthroughBlockKey{}); val != nil {
		b := val.(blockWithArgs)
		return b.block, b.args
	}
	return mlir.Block{}, nil
}

func newContextWithLabeledBlocks(ctx context.Context, block map[string]mlir.Block) context.Context {
	return context.WithValue(ctx, labeledBlocksKey{}, block)
}

func currentLabeledBlocks(ctx context.Context) map[string]mlir.Block {
	if val := ctx.Value(labeledBlocksKey{}); val != nil {
		return val.(map[string]mlir.Block)
	}
	return nil
}

func newContextWithInfo(ctx context.Context, info *types.Info) context.Context {
	return context.WithValue(ctx, infoKey{}, info)
}

func currentInfo(ctx context.Context) *types.Info {
	if val := ctx.Value(infoKey{}); val != nil {
		return val.(*types.Info)
	}
	return nil
}

func newContextWithTypeMap(ctx context.Context, typeMap TypeParamMap) context.Context {
	ctx = context.WithValue(ctx, typeMapKey{}, typeMap)
	// Create a per-instance type cache for TypeParam-containing types so that
	// different generic instances don't pollute each other's cache entries while
	// still allowing recursion-breaking writes within the same instance.
	ctx = context.WithValue(ctx, instanceTypeCacheKey{}, make(map[types.Type]mlir.TypeLike))
	// Create a processing set to detect recursion from TypeParam index collisions
	// across different generic scopes.
	ctx = context.WithValue(ctx, typeProcessingSetKey{}, make(map[types.Type]bool))
	return ctx
}

func currentTypeMap(ctx context.Context) TypeParamMap {
	if val := ctx.Value(typeMapKey{}); val != nil {
		return val.(TypeParamMap)
	}
	return nil
}

func currentInstanceTypeCache(ctx context.Context) map[types.Type]mlir.TypeLike {
	if val := ctx.Value(instanceTypeCacheKey{}); val != nil {
		return val.(map[types.Type]mlir.TypeLike)
	}
	return nil
}

func currentTypeProcessingSet(ctx context.Context) map[types.Type]bool {
	if val := ctx.Value(typeProcessingSetKey{}); val != nil {
		return val.(map[types.Type]bool)
	}
	return nil
}

func newContextWithScope(ctx context.Context, attribute goir.ScopeAttr) context.Context {
	return context.WithValue(ctx, scopeKey{}, attribute)
}

func currentScope(ctx context.Context) goir.ScopeAttr {
	if val := ctx.Value(scopeKey{}); val != nil {
		return val.(goir.ScopeAttr)
	}
	return goir.ScopeAttr{}
}
