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
		// Determine the element type of the slice
		elementType := c.createType(ctx, args[0].Type().Underlying().(*types.Slice).Elem())
		elementTypeDesc := c.createTypeDescriptor(ctx, elementType)
		appendArgs := [3]llvm.LLVMValueRef{c.addressOf(ctx, argValues[0]), nil, elementTypeDesc}
		sliceType := llvm.GetTypeByName2(c.currentContext(ctx), "slice")
		stringType := llvm.GetTypeByName2(c.currentContext(ctx), "string")
		if llvm.TypeIsEqual(llvm.TypeOf(argValues[1]), sliceType) {
			// Pass the slice directly to the runtime call
			appendArgs[1] = c.addressOf(ctx, argValues[1])
		} else if llvm.TypeIsEqual(stringType, llvm.TypeOf(argValues[1])) {
			// Convert the string to a slice
			appendArgs[1] = c.createSliceFromStringValue(ctx, argValues[1])
		} else {
			// Create a slice from the remaining arguments and pass that to the
			// runtime call.
			appendArgs[1] = c.addressOf(ctx, c.createSliceFromValues(ctx, argValues[1:].Ref(ctx)))
		}

		// Create the runtime call
		var sliceValue llvm.LLVMValueRef
		sliceValue, err = c.createRuntimeCall(ctx, "append", appendArgs[:])
		if err != nil {
			return nil, err
		}

		// Load the resulting slice
		value = llvm.BuildLoad2(c.builder, sliceType, c.addressOf(ctx, sliceValue), "")
	case "cap":
		switch args[0].Type().Underlying().(type) {
		case *types.Slice:
			value, err = c.createRuntimeCall(ctx, "sliceCap", []llvm.LLVMValueRef{c.addressOf(ctx, argValues[0])})
		case *types.Chan:
			panic("not implemented")
		default:
			panic("cap called with invalid value type")
		}
	case "close":
		panic("not implemented")
	case "complex":
		panic("not implemented")
	case "copy":
		elementType := c.createType(ctx, args[0].Type().Underlying().(*types.Slice).Elem())
		elementTypeDesc := c.createTypeDescriptor(ctx, elementType)
		copyArgs := [3]llvm.LLVMValueRef{c.addressOf(ctx, argValues[0]), nil, elementTypeDesc}
		stringType := llvm.GetTypeByName2(c.currentContext(ctx), "string")

		// If the second argument is a string, convert it to a byte slice
		if llvm.TypeIsEqual(stringType, llvm.TypeOf(argValues[1])) {
			copyArgs[1] = c.createSliceFromStringValue(ctx, argValues[1])
		} else {
			// Pass the slice directly
			copyArgs[1] = c.addressOf(ctx, argValues[1])
		}

		// Create the runtime call
		value, err = c.createRuntimeCall(ctx, "sliceCopy", copyArgs[:])
		if err != nil {
			return nil, err
		}
	case "delete":
		panic("not implemented")
	case "imag":
		panic("not implemented")
	case "len":
		switch t := args[0].Type().Underlying().(type) {
		case *types.Basic:
			switch t.Kind() {
			case types.String:
				value, err = c.createRuntimeCall(ctx, "stringLen", []llvm.LLVMValueRef{c.addressOf(ctx, argValues[0])})
			default:
				panic("len called with invalid value type")
			}
		case *types.Slice:
			value, err = c.createRuntimeCall(ctx, "sliceLen", []llvm.LLVMValueRef{c.addressOf(ctx, argValues[0])})
		case *types.Chan:
			panic("not implemented")
		default:
			panic("len called with invalid value type")
		}
	case "make":
		panic("unreachable")
	case "new":
		panic("unreachable")
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
