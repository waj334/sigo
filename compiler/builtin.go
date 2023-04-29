package compiler

import (
	"context"
	"go/types"
	"golang.org/x/tools/go/ssa"
	"omibyte.io/sigo/llvm"
)

func (c *Compiler) createBuiltinCall(ctx context.Context, builtin *ssa.Builtin, args []ssa.Value) (value llvm.LLVMValueRef, err error) {
	// Get the argument values
	argValues, err := c.createValues(ctx, args)
	if err != nil {
		return nil, err
	}

	// Create the proper builtin call based on the method and the type
	// NOTE: It MUST be defined in the runtime used by this application
	switch builtin.Name() {
	case "append":
		panic("not implemented")
	case "cap":
		panic("not implemented")
	case "close":
		panic("not implemented")
	case "complex":
		panic("not implemented")
	case "copy":
		panic("not implemented")
	case "delete":
		panic("not implemented")
	case "imag":
		panic("not implemented")
	case "len":
		switch t := args[0].Type().Underlying().(type) {
		case *types.Basic:
			switch t.Kind() {
			case types.String:
				value, err = c.createRuntimeCall(ctx, "stringLen", []llvm.LLVMValueRef{c.addressOf(argValues[0])})
			default:
				panic("len called with value of invalid kind")
			}
		case *types.Slice:
			value, err = c.createRuntimeCall(ctx, "sliceLen", []llvm.LLVMValueRef{c.addressOf(argValues[0])})
		default:
			panic("len called with value of non-basic type as argument")
		}
	case "make":
		panic("not implemented")
	case "new":
		panic("not implemented")
	case "panic":
		panic("not implemented")
	case "print", "println":
		// Create an array of interfaces to pass to the print runtime calls
		argType := llvm.GetTypeByName2(c.currentContext(ctx), "interface")
		if argType == nil {
			panic("missing \"interface\" type")
		}

		// Convert each argument into an interface
		var elements []llvm.LLVMValueRef
		for _, arg := range args {
			argValue, err := c.makeInterface(ctx, arg)
			if err != nil {
				return nil, err
			}
			elements = append(elements, argValue)
		}

		arg := llvm.BuildArrayAlloca(c.builder, argType, llvm.ConstInt(llvm.Int32TypeInContext(c.currentContext(ctx)), uint64(len(elements)), false), "")
		for i, _ := range args {
			index := llvm.ConstInt(llvm.Int32TypeInContext(c.currentContext(ctx)), uint64(i), false)
			addr := llvm.BuildGEP2(c.builder, argType, arg, []llvm.LLVMValueRef{index}, "")
			llvm.BuildStore(c.builder, elements[i], addr)
		}

		// Slice the array
		arg = c.createSlice(ctx, arg, argType, uint64(len(args)),
			llvm.ConstInt(llvm.Int32TypeInContext(c.currentContext(ctx)), 0, false),
			llvm.ConstInt(llvm.Int32TypeInContext(c.currentContext(ctx)), uint64(len(elements)), false),
			nil,
		)

		// Create a runtime call to either print or println
		value, err = c.createRuntimeCall(ctx, builtin.Name(), []llvm.LLVMValueRef{arg})
	case "real":
		panic("not implemented")
	case "recover":
		panic("not implemented")
	case "Add":
		// Add to pointer values
		value = llvm.BuildGEP2(c.builder, llvm.Int8TypeInContext(c.currentContext(ctx)), argValues[0], []llvm.LLVMValueRef{argValues[1]}, "")
	default:
		panic("encountered unknown builtin function")
	}

	return
}
