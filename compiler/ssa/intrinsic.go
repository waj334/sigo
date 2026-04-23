package ssa

import (
	"context"
	"fmt"
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
	"strings"

	"pkg.si-go.dev/go-mlir/mlir"
	"pkg.si-go.dev/sigo/goir/binding/goir"
)

func (b *Builder) isIntrinsic(ctx context.Context, expr *ast.CallExpr) bool {
	obj := b.objectOf(ctx, expr.Fun)
	switch obj := obj.(type) {
	case *types.Func:
		return isIntrinsic(qualifiedFuncName(obj))
	default:
		return false
	}
}

func isIntrinsic(symbol string) bool {
	switch symbol {
	case
		"internal/abi.EscapeNonString",

		"sync/atomic.AddUint32",
		"sync/atomic.AddInt32",
		"sync/atomic.AddUint64",
		"sync/atomic.AddInt64",
		"sync/atomic.AddUintptr",

		"sync/atomic.LoadUint32",
		"sync/atomic.LoadInt32",
		"sync/atomic.LoadUint64",
		"sync/atomic.LoadInt64",
		"sync/atomic.LoadUintptr",
		"sync/atomic.LoadPointer",

		"sync/atomic.StoreUint32",
		"sync/atomic.StoreInt32",
		"sync/atomic.StoreUint64",
		"sync/atomic.StoreInt64",
		"sync/atomic.StoreUintptr",
		"sync/atomic.StorePointer",

		"sync/atomic.SwapUint32",
		"sync/atomic.SwapInt32",
		"sync/atomic.SwapUint64",
		"sync/atomic.SwapInt64",
		"sync/atomic.SwapUintptr",
		"sync/atomic.SwapPointer",

		"sync/atomic.CompareAndSwapUint32",
		"sync/atomic.CompareAndSwapInt32",
		"sync/atomic.CompareAndSwapUint64",
		"sync/atomic.CompareAndSwapInt64",
		"sync/atomic.CompareAndSwapUintptr",
		"sync/atomic.CompareAndSwapPointer",

		"volatile.LoadInt8",
		"volatile.LoadInt16",
		"volatile.LoadInt32",
		"volatile.LoadInt64",
		"volatile.LoadUint8",
		"volatile.LoadUint16",
		"volatile.LoadUint32",
		"volatile.LoadUint64",
		"volatile.LoadUintptr",
		"volatile.LoadPointer",

		"volatile.StoreInt8",
		"volatile.StoreInt16",
		"volatile.StoreInt32",
		"volatile.StoreInt64",
		"volatile.StoreUint8",
		"volatile.StoreUint16",
		"volatile.StoreUint32",
		"volatile.StoreUint64",
		"volatile.StoreUintptr",
		"volatile.StorePointer",

		"asm.In",
		"asm.Out",
		"asm.InOut",
		"Asm.Clobber",
		"asm.Inline",

		"nonstandard.PointerOf":
		return true
	default:
		return false
	}
}

func (b *Builder) emitIntrinsic(ctx context.Context, expr *ast.CallExpr) []mlir.ValueLike {
	F := b.objectOf(ctx, expr.Fun).(*types.Func)
	signature := F.Type().Underlying().(*types.Signature)
	symbol := qualifiedFuncName(F)
	location := b.location(ctx, expr.Pos())

	switch symbol {
	case "internal/abi.EscapeNonString":
		addr := b.addressOf(ctx, expr.Args[0], location)
		result, _ := addr.AsResult()
		goir.AllocaOperationSetIsHeap(result.OwningOperation(), true)
		return nil
	case "sync/atomic.AddUint32", "sync/atomic.AddInt32", "sync/atomic.AddUint64", "sync/atomic.AddInt64", "sync/atomic.AddUintptr":
		args := b.emitCallArgs2(ctx, expr.Args)
		T := b.GetType(ctx, signature.Results().At(0).Type())
		op := goir.NewAtomicAddIOperation(b.ctx, T, args[0], args[1], location)
		appendOperation(ctx, op)
		return resultsOf(op)
	case "sync/atomic.CompareAndSwapUint32", "sync/atomic.CompareAndSwapInt32", "sync/atomic.CompareAndSwapUint64", "sync/atomic.CompareAndSwapInt64", "sync/atomic.CompareAndSwapUintptr", "sync/atomic.CompareAndSwapPointer":
		args := b.emitCallArgs2(ctx, expr.Args)
		T := b.GetType(ctx, signature.Results().At(0).Type())
		op := goir.NewAtomicCompareAndSwapOperation(b.ctx, T, args[0], args[1], args[2], location)
		appendOperation(ctx, op)
		return resultsOf(op)
	case "sync/atomic.LoadUint32", "sync/atomic.LoadInt32", "sync/atomic.LoadUint64", "sync/atomic.LoadInt64", "sync/atomic.LoadUintptr", "sync/atomic.LoadPointer":
		args := b.emitCallArgs2(ctx, expr.Args)
		T := b.GetType(ctx, signature.Results().At(0).Type())
		op := goir.NewAtomicLoadOperation(b.ctx, args[0], T, location)
		appendOperation(ctx, op)
		return resultsOf(op)
	case "sync/atomic.StoreUint32", "sync/atomic.StoreInt32", "sync/atomic.StoreUint64", "sync/atomic.StoreInt64", "sync/atomic.StoreUintptr", "sync/atomic.StorePointer":
		args := b.emitCallArgs2(ctx, expr.Args)
		op := goir.NewAtomicStoreOperation(b.ctx, args[1], args[0], location)
		appendOperation(ctx, op)
		return resultsOf(op)
	case "sync/atomic.SwapUint32", "sync/atomic.SwapInt32", "sync/atomic.SwapUint64", "sync/atomic.SwapInt64", "sync/atomic.SwapUintptr", "sync/atomic.SwapPointer":
		args := b.emitCallArgs2(ctx, expr.Args)
		T := b.GetType(ctx, signature.Results().At(0).Type())
		op := goir.NewAtomicSwapOperation(b.ctx, T, args[0], args[1], location)
		appendOperation(ctx, op)
		return resultsOf(op)
	case "volatile.LoadInt8", "volatile.LoadInt16", "volatile.LoadInt32", "volatile.LoadInt64", "volatile.LoadUint8", "volatile.LoadUint16", "volatile.LoadUint32", "volatile.LoadUint64", "volatile.LoadUintptr", "volatile.LoadPointer":
		args := b.emitCallArgs2(ctx, expr.Args)
		T := b.GetType(ctx, signature.Results().At(0).Type())
		op := goir.NewVolatileLoadOperation(b.ctx, args[0], T, location)
		appendOperation(ctx, op)
		return resultsOf(op)
	case "volatile.StoreInt8", "volatile.StoreInt16", "volatile.StoreInt32", "volatile.StoreInt64", "volatile.StoreUint8", "volatile.StoreUint16", "volatile.StoreUint32", "volatile.StoreUint64", "volatile.StoreUintptr", "volatile.StorePointer":
		args := b.emitCallArgs2(ctx, expr.Args)
		op := goir.NewVolatileStoreOperation(b.ctx, args[1], args[0], location)
		appendOperation(ctx, op)
		return resultsOf(op)
	case "asm.In", "asm.Out", "asm.InOut", "asm.Clobber":
		panic("unreachable")
	case "asm.Inline":
		b.emitInlineAssembly(ctx, expr)
		return nil
	case "nonstandard.PointerOf":
		obj := b.objectOf(ctx, expr.Args[0]).(*types.Func)
		symbolName := qualifiedFuncName(obj)
		value := b.addressOfSymbol(ctx, symbolName, b.ptr, location)
		return b.values(value)
	default:
		panic("unhandled")
	}
}

func (b *Builder) emitInlineAssembly(ctx context.Context, expr *ast.CallExpr) {
	location := b.location(ctx, expr.Pos())
	constAsmStr, ok := expr.Args[0].(*ast.BasicLit)
	if !ok {
		panic("TODO: handle error")
	}

	if constAsmStr.Kind != token.STRING {
		panic("TODO: handle invalid asm string type error")
	}

	var operands []mlir.ValueLike
	var clobberRegisters []mlir.AttributeLike
	var constraintAttrs []mlir.AttributeLike

	args := expr.Args[1:]
	for _, arg := range args {
		switch arg := arg.(type) {
		case *ast.CallExpr:
			obj := b.objectOf(ctx, arg.Fun)
			switch obj := obj.(type) {
			case *types.Func:
				switch obj.Name() {
				case "Out", "In", "InOut":
					class := "r"
					var operandName string
					var operandValue mlir.Value
					earlyClobber := false

					// Process call arguments.
					for _, argExpr := range arg.Args {
						switch argExpr := argExpr.(type) {
						case *ast.SelectorExpr:
							obj := b.objectOf(ctx, argExpr.Sel)
							switch obj := obj.(type) {
							case *types.Const:
								if T, ok := obj.Type().(*types.Named); ok {
									typeName := qualifiedName(T.Obj().Name(), T.Obj().Pkg())
									switch typeName {
									case "asm.RegisterClass":
										class = strings.TrimSpace(constant.StringVal(obj.Val()))
									case "asm.clobber":
										switch argExpr.Sel.Name {
										case "Reserve":
											earlyClobber = true
										default:
											panic("unhandled")
										}
									}
								}
							case *types.Var:
								operandValue = b.emitExpr(ctx, argExpr)[0].AsValue()
							}
						case *ast.CallExpr:
							obj := b.objectOf(ctx, argExpr.Fun)
							switch qualifiedName(obj.Name(), obj.Pkg()) {
							case "asm.Alias":
								lit := argExpr.Args[0].(*ast.BasicLit)
								operandName = strings.TrimSpace(cleanConstString(lit.Value))
							case "asm.RegisterClass":
								lit := argExpr.Args[0].(*ast.BasicLit)
								class = strings.TrimSpace(cleanConstString(lit.Value))
							default:
								panic("unhandled")
							}
						case *ast.UnaryExpr:
							ident := argExpr.X.(*ast.Ident)
							if len(operandName) == 0 {
								operandName = ident.Name
							}
							operandValue = b.emitExpr(ctx, argExpr)[0].AsValue()
						case *ast.Ident:
							if len(operandName) == 0 {
								operandName = argExpr.Name
							}
							operandValue = b.emitExpr(ctx, argExpr)[0].AsValue()
						default:
							panic("unhandled")
						}
					}

					operandIndex := -1
					if !operandValue.IsNull() {
						operandIndex = len(operands)
						operands = append(operands, operandValue)
					}

					switch obj.Name() {
					case "InOut":
						constraintAttrs = append(constraintAttrs,
							goir.NewAsmConstraintAttr(b.ctx, class, goir.AsmConstraintDirectionInOut, operandName, operandIndex, earlyClobber))
					case "In":
						constraintAttrs = append(constraintAttrs,
							goir.NewAsmConstraintAttr(b.ctx, class, goir.AsmConstraintDirectionIn, operandName, operandIndex, false))
					case "Out":
						constraintAttrs = append(constraintAttrs,
							goir.NewAsmConstraintAttr(b.ctx, class, goir.AsmConstraintDirectionOut, operandName, operandIndex, earlyClobber))
					}
				case "Clobber":
					switch reg := arg.Args[0].(type) {
					case *ast.CallExpr:
						lit := reg.Args[0].(*ast.BasicLit)
						clobberRegisters = append(clobberRegisters,
							mlir.NewStringAttr(b.ctx, cleanConstString(lit.Value)))
					case *ast.SelectorExpr:
						obj := b.objectOf(ctx, reg.Sel).(*types.Const)
						clobberRegisters = append(clobberRegisters,
							mlir.NewStringAttr(b.ctx, cleanConstString(constant.StringVal(obj.Val()))))
					}
				default:
					panic("unhandled")
				}
			}
		default:
			panic("TODO: handle invalid operand type error")
		}
	}

	asmStr := cleanConstString(constAsmStr.Value)
	asmStrAttr := mlir.NewStringAttr(b.ctx, asmStr)

	// NOTE: Output operands must appear first then input operands.
	op := goir.NewInlineAssemblyOperation(b.ctx, asmStrAttr, constraintAttrs, clobberRegisters, operands, location)
	appendOperation(ctx, op)
}

func formatInputOutputConstraint(register string) string {
	register = strings.TrimSpace(register)
	return fmt.Sprintf("*%s", register)
}

func formatOutputConstraint(register string, earlyClobber bool) string {
	register = strings.TrimSpace(register)
	if earlyClobber {
		return fmt.Sprintf("=&%s", register)
	}
	return fmt.Sprintf("=%s", register)
}

func formatClobberConstraint(register string) string {
	register = strings.TrimSpace(register)
	return fmt.Sprintf("~{%s}", register)
}
