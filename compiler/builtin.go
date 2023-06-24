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
		// Determine the element type of the slice
		elementType := c.createType(ctx, args[0].Type().Underlying().(*types.Slice).Elem())
		elementTypeDesc := c.createTypeDescriptor(ctx, elementType)
		appendArgs := [3]llvm.LLVMValueRef{c.addressOf(ctx, argValues[0].UnderlyingValue(ctx)), nil, elementTypeDesc}
		sliceType := c.createRuntimeType(ctx, "_slice").valueType
		stringType := c.createRuntimeType(ctx, "_string").valueType
		if llvm.TypeIsEqual(llvm.TypeOf(argValues[1].UnderlyingValue(ctx)), sliceType) {
			// Pass the slice directly to the runtime call
			appendArgs[1] = c.addressOf(ctx, argValues[1].UnderlyingValue(ctx))
		} else if llvm.TypeIsEqual(stringType, llvm.TypeOf(argValues[1].UnderlyingValue(ctx))) {
			// Convert the string to a slice
			appendArgs[1] = c.createSliceFromStringValue(ctx, argValues[1].UnderlyingValue(ctx))
		} else {
			// Create a slice from the remaining arguments and pass that to the
			// runtime call.
			appendArgs[1] = c.addressOf(ctx, c.createSliceFromValues(ctx, argValues[1:].Ref(ctx)))
		}

		// Create the runtime call
		var sliceValue llvm.LLVMValueRef
		sliceValue = c.createRuntimeCall(ctx, "sliceAppend", appendArgs[:])

		// Load the resulting slice
		value = llvm.BuildLoad2(c.builder, sliceType, c.addressOf(ctx, sliceValue), "")
	case "cap":
		switch args[0].Type().Underlying().(type) {
		case *types.Slice:
			value = c.createRuntimeCall(ctx, "sliceCap", []llvm.LLVMValueRef{c.addressOf(ctx, argValues[0].UnderlyingValue(ctx))})
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
		copyArgs := [3]llvm.LLVMValueRef{c.addressOf(ctx, argValues[0].UnderlyingValue(ctx)), nil, elementTypeDesc}
		stringType := c.createRuntimeType(ctx, "_string").valueType

		// If the second argument is a string, convert it to a byte slice
		if llvm.TypeIsEqual(stringType, llvm.TypeOf(argValues[1].UnderlyingValue(ctx))) {
			copyArgs[1] = c.createSliceFromStringValue(ctx, argValues[1].UnderlyingValue(ctx))
		} else {
			// Pass the slice directly
			copyArgs[1] = c.addressOf(ctx, argValues[1].UnderlyingValue(ctx))
		}

		// Create the runtime call
		value = c.createRuntimeCall(ctx, "sliceCopy", copyArgs[:])
	case "delete":
		value = c.createRuntimeCall(ctx, "mapDelete", []llvm.LLVMValueRef{
			c.addressOf(ctx, argValues[0].UnderlyingValue(ctx)),
			c.addressOf(ctx, argValues[1].UnderlyingValue(ctx)),
		})
	case "imag":
		panic("not implemented")
	case "len":
		switch t := args[0].Type().Underlying().(type) {
		case *types.Basic:
			switch t.Kind() {
			case types.String:
				value = c.createRuntimeCall(ctx, "stringLen", []llvm.LLVMValueRef{c.addressOf(ctx, argValues[0].UnderlyingValue(ctx))})
			default:
				panic("len called with invalid value type")
			}
		case *types.Slice:
			value = c.createRuntimeCall(ctx, "sliceLen", []llvm.LLVMValueRef{c.addressOf(ctx, argValues[0].UnderlyingValue(ctx))})
		case *types.Chan:
			panic("not implemented")
		default:
			panic("len called with invalid value type")
		}
	case "print", "println":
		// Create an array of interfaces to pass to the print runtime calls
		argType := c.createRuntimeType(ctx, "_interface").valueType
		if argType == nil {
			panic("missing \"interface\" type")
		}

		// Convert each argument into an interface
		var elements []llvm.LLVMValueRef
		for _, arg := range args {
			argValue := c.makeInterface(ctx, arg)
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
		value = c.createRuntimeCall(ctx, "_"+builtin.Name(), []llvm.LLVMValueRef{arg})
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
