package compiler

import (
	"context"
	"go/types"
	"golang.org/x/tools/go/ssa"
	"omibyte.io/sigo/llvm"
)

func (c *Compiler) createBuiltinCall(ctx context.Context, builtin *ssa.Builtin, args []ssa.Value) (value llvm.LLVMValueRef) {
	// Get the argument values
	argValues := c.createValues(ctx, args)

	// Create the proper builtin call based on the method and the type
	// NOTE: It MUST be defined in the runtime used by this application
	switch builtin.Name() {
	case "append":
		argValues := argValues.Ref(ctx)
		sliceType := args[0].Type().Underlying().(*types.Slice)
		elementType := c.createTypeDescriptor(ctx, c.createType(ctx, sliceType.Elem()))
		value = c.createRuntimeCall(ctx, "sliceAppend", []llvm.LLVMValueRef{
			argValues[0], argValues[1], elementType,
		})
	case "cap":
		switch args[0].Type().Underlying().(type) {
		case *types.Slice:
			value = c.createRuntimeCall(ctx, "sliceCap", []llvm.LLVMValueRef{argValues[0].UnderlyingValue(ctx)})
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
		argValues := argValues.Ref(ctx)
		elementType := c.createType(ctx, args[0].Type().Underlying().(*types.Slice).Elem())
		elementTypeDesc := c.createTypeDescriptor(ctx, elementType)
		copyArgs := [3]llvm.LLVMValueRef{argValues[0], nil, elementTypeDesc}
		stringType := c.createRuntimeType(ctx, "_string").valueType

		// If the second argument is a string, convert it to a byte slice
		if llvm.TypeIsEqual(stringType, llvm.TypeOf(argValues[1])) {
			copyArgs[1] = c.createSliceFromStringValue(ctx, argValues[1])
		} else {
			// Pass the slice directly
			copyArgs[1] = argValues[1]
		}

		// Create the runtime call
		value = c.createRuntimeCall(ctx, "sliceCopy", copyArgs[:])
	case "delete":
		argValues := argValues.Ref(ctx)
		value = c.createRuntimeCall(ctx, "mapDelete", argValues)
	case "imag":
		panic("not implemented")
	case "len":
		switch t := args[0].Type().Underlying().(type) {
		case *types.Basic:
			switch t.Kind() {
			case types.String:
				value = c.createRuntimeCall(ctx, "stringLen", []llvm.LLVMValueRef{argValues[0].UnderlyingValue(ctx)})
			default:
				panic("len called with invalid value type")
			}
		case *types.Slice:
			value = c.createRuntimeCall(ctx, "sliceLen", []llvm.LLVMValueRef{argValues[0].UnderlyingValue(ctx)})
		case *types.Chan:
			panic("not implemented")
		default:
			panic("len called with invalid value type")
		}
	case "print", "println":
		// Create an array of interfaces to pass to the print runtime calls
		argType := c.createRuntimeType(ctx, "_interface").valueType
		if argType == nil {
			panic(`missing "interface" type`)
		}

		// Convert each argument into an interface
		var elements []llvm.LLVMValueRef
		for _, arg := range args {
			argValue := c.makeInterface(ctx, arg)
			elements = append(elements, argValue)
		}

		arr := c.createArrayAlloca(ctx, argType, uint64(len(elements)), "")
		for i, _ := range args {
			index := llvm.ConstInt(llvm.Int32TypeInContext(c.currentContext(ctx)), uint64(i), false)
			addr := llvm.BuildGEP2(c.builder, argType, arr, []llvm.LLVMValueRef{index}, "")
			llvm.BuildStore(c.builder, elements[i], addr)
		}

		// Create the print args slice
		sliceType := c.createRuntimeType(ctx, "_slice")
		argsSlice := llvm.BuildInsertValue(c.builder, sliceType.Undef(), arr, 0, "")
		argsSlice = llvm.BuildInsertValue(c.builder, argsSlice, llvm.ConstInt(c.int32Type(ctx), uint64(len(elements)), false), 1, "")
		argsSlice = llvm.BuildInsertValue(c.builder, argsSlice, llvm.ConstInt(c.int32Type(ctx), uint64(len(elements)), false), 2, "")

		// Create a runtime call to either print or println
		value = c.createRuntimeCall(ctx, "_"+builtin.Name(), []llvm.LLVMValueRef{argsSlice})
	case "real":
		panic("not implemented")
	case "recover":
		value = c.createRuntimeCall(ctx, "_recover", nil)
	case "Add":
		// Add to pointer values
		value = llvm.BuildGEP2(c.builder, llvm.Int8TypeInContext(c.currentContext(ctx)), argValues[0].UnderlyingValue(ctx), []llvm.LLVMValueRef{argValues[1].UnderlyingValue(ctx)}, "")
	default:
		panic("encountered unknown builtin function")
	}

	return
}
