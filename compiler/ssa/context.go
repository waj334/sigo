package ssa

import (
	"context"
	"go/types"
	"omibyte.io/sigo/mlir"
)

type (
	regionKey            struct{}
	blockKey             struct{}
	successorBlockKey    struct{}
	predecessorBlockKey  struct{}
	labeledBlocksKey     struct{}
	identifierKey        struct{}
	identifierPostfixKey struct{}
	lhsListKey           struct{}
	rhsIndexKey          struct{}
	funcDataKey          struct{}
	globalKey            struct{}
	jobQueueKey          struct{}
	infoKey              struct{}
)

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
	return nil
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
	return nil
}

func setCurrentBlock(ctx context.Context, target mlir.Block) {
	if val := ctx.Value(blockKey{}); val != nil {
		blockPtr := val.(**mlir.Block)
		*blockPtr = &target
		return
	}
	panic("No block pointer in context")
}

func newContextWithSuccessorBlock(ctx context.Context, block mlir.Block) context.Context {
	return context.WithValue(ctx, successorBlockKey{}, block)
}

func currentSuccessorBlock(ctx context.Context) mlir.Block {
	if val := ctx.Value(successorBlockKey{}); val != nil {
		return val.(mlir.Block)
	}
	return nil
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

func newContextWithPredecessorBlock(ctx context.Context, block mlir.Block) context.Context {
	return context.WithValue(ctx, predecessorBlockKey{}, block)
}

func currentPredecessorBlock(ctx context.Context) mlir.Block {
	if val := ctx.Value(predecessorBlockKey{}); val != nil {
		return val.(mlir.Block)
	}
	return nil
}

func newContextWithLhsList(ctx context.Context, lhs []types.Type) context.Context {
	// Enforce that no type is untyped.
	for _, T := range lhs {
		if T != nil && typeHasFlags(T, types.IsUntyped) {
			panic("untyped type are forbidden")
		}
	}
	return context.WithValue(ctx, lhsListKey{}, lhs)
}

func currentLhsList(ctx context.Context) []types.Type {
	if val := ctx.Value(lhsListKey{}); val != nil {
		return val.([]types.Type)
	}
	return nil
}

func newContextWithRhsIndex(ctx context.Context, rhs int) context.Context {
	return context.WithValue(ctx, rhsIndexKey{}, rhs)
}

func currentRhsIndex(ctx context.Context) int {
	if val := ctx.Value(rhsIndexKey{}); val != nil {
		return val.(int)
	}
	return -1
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
