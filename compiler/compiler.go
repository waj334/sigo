package compiler

import (
	"context"
	"fmt"
	"go/constant"
	"go/token"
	"go/types"
	"golang.org/x/tools/go/ssa"
	"path/filepath"

	"omibyte.io/sigo/llvm"
)

type (
	llvmContextKey     struct{}
	currentPackageKey  struct{}
	entryBlockKey      struct{}
	currentBlockKey    struct{}
	currentFnTypeKey   struct{}
	currentScopeKey    struct{}
	currentLocationKey struct{}
)

type Function struct {
	value    llvm.LLVMValueRef
	llvmType llvm.LLVMTypeRef
	def      *ssa.Function
	diFile   llvm.LLVMMetadataRef
	compiled bool
}

type Variable struct {
	value llvm.LLVMValueRef
	dbg   llvm.LLVMMetadataRef
}

type Type struct {
	valueType  llvm.LLVMTypeRef
	debugType  llvm.LLVMMetadataRef
	name       string
	spec       types.Type
	descriptor llvm.LLVMValueRef
}

type Phi struct {
	value llvm.LLVMValueRef
	edges []ssa.Value
}

type Location struct {
	file     llvm.LLVMMetadataRef
	line     llvm.LLVMMetadataRef
	position token.Position
}

type Compiler struct {
	options     Options
	module      llvm.LLVMModuleRef
	builder     llvm.LLVMBuilderRef
	dibuilder   llvm.LLVMDIBuilderRef
	functions   map[*types.Signature]*Function
	types       map[types.Type]Type
	descriptors map[types.Type]llvm.LLVMValueRef
	values      map[ssa.Value]llvm.LLVMValueRef
	variables   map[llvm.LLVMValueRef]*Variable
	phis        map[*ssa.Phi]Phi
	blocks      map[*ssa.BasicBlock]llvm.LLVMBasicBlockRef

	uintptrType Type
	ptrType     Type
}

func NewCompiler(options Options) *Compiler {
	// Create the LLVM context
	ctx := llvm.ContextCreate()

	// Create the module
	module := llvm.ModuleCreateWithNameInContext("main", ctx)

	// Set the data layout for this module from the target
	llvm.SetDataLayout(module, llvm.CopyStringRepOfTargetData(options.Target.dataLayout))

	// Create the instruction builder
	builder := llvm.CreateBuilderInContext(ctx)

	// Create the DIBuilder
	dibuilder := llvm.CreateDIBuilder(module)

	// Create the uintptr type based on the machine's pointer width
	uintptrType := Type{
		valueType: llvm.IntPtrTypeInContext(ctx, options.Target.dataLayout),
	}
	ptrType := Type{
		valueType: llvm.PointerType(uintptrType.valueType, 0),
	}

	// Create and return the compiler
	return &Compiler{
		options:     options,
		module:      module,
		builder:     builder,
		dibuilder:   dibuilder,
		functions:   map[*types.Signature]*Function{},
		types:       map[types.Type]Type{},
		descriptors: map[types.Type]llvm.LLVMValueRef{},
		values:      map[ssa.Value]llvm.LLVMValueRef{},
		variables:   map[llvm.LLVMValueRef]*Variable{},
		blocks:      map[*ssa.BasicBlock]llvm.LLVMBasicBlockRef{},
		phis:        map[*ssa.Phi]Phi{},
		uintptrType: uintptrType,
		ptrType:     ptrType,
	}
}

func (c *Compiler) Module() llvm.LLVMModuleRef {
	return c.module
}

func (c *Compiler) Dispose() {
	llvm.DisposeDIBuilder(c.dibuilder)
	llvm.DisposeBuilder(c.builder)
	llvm.DisposeModule(c.module)
}

func (c *Compiler) currentContext(ctx context.Context) llvm.LLVMContextRef {
	if v := ctx.Value(llvmContextKey{}); v != nil {
		return v.(llvm.LLVMContextRef)
	}
	return nil
}

func (c *Compiler) currentEntryBlock(ctx context.Context) llvm.LLVMBasicBlockRef {
	if v := ctx.Value(entryBlockKey{}); v != nil {
		return v.(llvm.LLVMBasicBlockRef)
	}
	return nil
}

func (c *Compiler) currentBlock(ctx context.Context) llvm.LLVMBasicBlockRef {
	if v := ctx.Value(currentBlockKey{}); v != nil {
		return v.(llvm.LLVMBasicBlockRef)
	}
	return nil
}

func (c *Compiler) currentPackage(ctx context.Context) *ssa.Package {
	if v := ctx.Value(currentPackageKey{}); v != nil {
		return v.(*ssa.Package)
	}
	return nil
}

func (c *Compiler) currentFunction(ctx context.Context) *Function {
	if v := ctx.Value(currentFnTypeKey{}); v != nil {
		return v.(*Function)
	}
	return nil
}

func (c *Compiler) currentScope(ctx context.Context) llvm.LLVMMetadataRef {
	if v := ctx.Value(currentScopeKey{}); v != nil {
		return v.(llvm.LLVMMetadataRef)
	}
	return nil
}

func (c *Compiler) currentLocation(ctx context.Context) Location {
	return ctx.Value(currentLocationKey{}).(Location)
}

func (c *Compiler) offset(index int) int {
	// Calculate the offset using the word size of the target machine
	return 0 //llvm.SizeOf(c.ptrType)
}

func (c *Compiler) CompilePackage(ctx context.Context, pkg *ssa.Package) error {
	// Set the LLVM context to the compiler context
	ctx = context.WithValue(ctx, llvmContextKey{}, llvm.GetGlobalContext())

	// Set the current package in the context
	ctx = context.WithValue(ctx, currentPackageKey{}, pkg)

	if c.options.GenerateDebugInfo {
		size := llvm.StoreSizeOfType(c.options.Target.dataLayout, c.uintptrType.valueType) * 8

		// Create the debug info for the pointer type
		c.uintptrType.debugType = llvm.DIBuilderCreateBasicType(
			c.dibuilder, "uintptr", uint64(size), DW_ATE_unsigned, 0)

		c.ptrType.debugType = llvm.DIBuilderCreateBasicType(c.dibuilder, "unsafe.Pointer", size, DW_ATE_address, 0)
	}

	// Create any named types first
	for _, member := range pkg.Members {
		if ssaType, ok := member.(*ssa.Type); ok {
			c.createType(ctx, ssaType.Type())
		}
	}

	// Create constants and globals next
	for _, member := range pkg.Members {
		switch member := member.(type) {
		case *ssa.NamedConst:
			if _, err := c.createExpression(ctx, member.Value); err != nil {
				return err
			}
		case *ssa.Global:
			if _, err := c.createExpression(ctx, member); err != nil {
				return err
			}
		}
	}

	var uncompiledFunctions []*Function

	// Create the type for each function first so that they are available when it's time to create the blocks.
	for _, member := range pkg.Members {
		if ssaFn, ok := member.(*ssa.Function); ok {
			fn, err := c.createFunction(ctx, ssaFn)
			if err != nil {
				return err
			}
			c.functions[ssaFn.Signature] = fn
			uncompiledFunctions = append(uncompiledFunctions, fn)
		}
	}

	// Finally create the blocks for each function
	for _, fn := range uncompiledFunctions {
		// Only attempt to compile functions that have blocks. This will allow
		// the compiler to support forward declarations of functions.
		if len(fn.def.Blocks) > 0 {
			if err := c.createFunctionBlocks(ctx, fn); err != nil {
				return err
			}
		}
	}

	return nil
}

func (c *Compiler) createFunction(ctx context.Context, fn *ssa.Function) (*Function, error) {
	var returnValueTypes []llvm.LLVMTypeRef
	var argValueTypes []llvm.LLVMTypeRef
	var argDiTypes []llvm.LLVMMetadataRef
	var returnType llvm.LLVMTypeRef

	// Determine the function name. //go:linkname can override this
	name := fmt.Sprint(fn.Pkg.Pkg.Path(), ".", fn.Name())
	if linkName, ok := c.options.LinkNames[name]; ok {
		// Attempt to find the existing function with the same linkname
		if fnValue := llvm.GetNamedFunction(c.module, linkName); fnValue != nil {
			// Find the function struct with the matching value
			for _, existingFn := range c.functions {
				if existingFn.value == fnValue {
					// Return this function directly so that they are "linked".
					return existingFn, nil
				}
			}
		}
		// Override the generated name
		name = linkName
	}

	if numArgs := fn.Signature.Results().Len(); numArgs == 0 {
		returnType = llvm.VoidTypeInContext(c.currentContext(ctx))
	} else if numArgs == 1 {
		returnType = c.createType(ctx, fn.Signature.Results().At(0).Type()).valueType
	} else {
		// Create a struct type to store the return values into
		for i := 0; i < numArgs; i++ {
			typ := fn.Signature.Results().At(i).Type()
			returnValueTypes = append(returnValueTypes, c.createType(ctx, typ).valueType)
		}
		returnType = llvm.StructTypeInContext(c.currentContext(ctx), returnValueTypes, false)
	}

	// Create types for the arguments
	for _, arg := range fn.Params {
		typ := c.createType(ctx, arg.Type())
		argValueTypes = append(argValueTypes, typ.valueType)
		argDiTypes = append(argDiTypes, typ.debugType)
	}

	// Create the function type
	fnType := llvm.FunctionType(returnType, argValueTypes, fn.Signature.Variadic())

	// Add the function to the current module
	fnValue := llvm.AddFunction(c.module, name, fnType)

	result := Function{
		value:    fnValue,
		llvmType: fnType,
		def:      fn,
	}

	// Create the file information for this function
	if c.options.GenerateDebugInfo {
		file := fn.Prog.Fset.File(fn.Pos())
		filename := fn.Pkg.Pkg.Name()
		line := uint(0)

		// Some functions, like package initializers, do not have position information
		if file != nil {
			// Extract the file info
			filename = file.Name()
			line = uint(file.Line(fn.Pos()))
		}

		result.diFile = llvm.DIBuilderCreateFile(
			c.dibuilder,
			filepath.Base(filename),
			filepath.Dir(filename))

		// Create the debug information for this function
		subType := llvm.DIBuilderCreateSubroutineType(
			c.dibuilder,
			result.diFile,
			argDiTypes,
			0)

		subprogram := llvm.DIBuilderCreateFunction(
			c.dibuilder,
			nil,
			fn.Name(),
			fn.Name(), // This should probably be the fully qualified package naming
			result.diFile,
			line,
			subType,
			true,
			true, 0, 0, false)

		// Apply this metadata to the function
		llvm.SetSubprogram(fnValue, subprogram)
	}

	return &result, nil
}

func (c *Compiler) createFunctionBlocks(ctx context.Context, fn *Function) error {
	// Panic now if this function was already compiled
	if fn.compiled {
		panic(fmt.Sprintf("multiple definitions of function \"%s\" exist", fn.def.Object().Id()))
	}

	c.println(Debug, "Compiling", fn.def.Name())
	defer c.println(Debug, "Done compiling", fn.def.Name())

	// Set the current function type in the context
	ctx = context.WithValue(ctx, currentFnTypeKey{}, fn)

	var file *token.File
	var scope llvm.LLVMMetadataRef

	if c.options.GenerateDebugInfo {
		// Get the file information for this function
		file = fn.def.Prog.Fset.File(fn.def.Pos())

		// Create the scope debug information for this function
		scope = llvm.GetSubprogram(fn.value)
		ctx = context.WithValue(ctx, currentScopeKey{}, scope)
	}

	// Create all blocks first so that branches can be made during instruction
	// creation.
	for _, block := range fn.def.Blocks {
		// Create a new block
		bb := llvm.AppendBasicBlockInContext(c.currentContext(ctx), fn.value, fn.def.Name()+"."+block.Comment)
		c.blocks[block] = bb
		if block.Comment == "entry" {
			// Set the current entry block in the context
			ctx = context.WithValue(ctx, entryBlockKey{}, bb)
		}
	}

	// Create the instructions in each of the function's blocks
	for i, block := range fn.def.Blocks {
		insertionBlock, ok := c.blocks[block]
		if !ok {
			panic("block not created")
		}

		c.println(Debug, "Processing block", i, "-", block.Comment)

		// All further instructions should go into this block
		llvm.PositionBuilderAtEnd(c.builder, insertionBlock)

		// Set the current block in the context
		ctx = context.WithValue(ctx, currentBlockKey{}, insertionBlock)

		// Create each instruction in the block
		for _, instr := range block.Instrs {
			if c.options.GenerateDebugInfo {
				// Get the location information for this instruction
				locationInfo := file.Position(instr.Pos())

				// Create the file debug information
				location := Location{
					file: fn.diFile,
					line: llvm.DIBuilderCreateDebugLocation(
						c.currentContext(ctx),
						uint(locationInfo.Line),
						uint(locationInfo.Column),
						scope,
						nil,
					),
					position: locationInfo,
				}

				// Set the current location in the context
				ctx = context.WithValue(ctx, currentLocationKey{}, location)
			}

			// Create the instruction
			if _, err := c.createInstruction(ctx, instr); err != nil {
				return err
			}
		}

		c.println(Debug, "Done processing block", i)
	}

	// Mark this function as compiled
	fn.compiled = true

	return nil
}

func (c *Compiler) createInstruction(ctx context.Context, instr ssa.Instruction) (result llvm.LLVMValueRef, err error) {
	c.printf(Debug, "Processing instruction %T: %s\n", instr, instr.String())

	// Create the specific instruction
	switch instr := instr.(type) {
	case *ssa.Defer:
		panic("not implemented")
	case *ssa.Go:
		panic("not implemented")
	case *ssa.If:
		condValue, err := c.createExpression(ctx, instr.Cond)
		if err != nil {
			return nil, err
		}

		b0 := c.blocks[instr.Block().Succs[0]]
		b1 := c.blocks[instr.Block().Succs[1]]

		result = llvm.BuildCondBr(c.builder, condValue, b0, b1)
	case *ssa.Jump:
		if block, ok := c.blocks[instr.Block()]; ok {
			result = llvm.BuildBr(c.builder, block)
		} else {
			panic("block not created")
		}
	case *ssa.MapUpdate:
		panic("not implemented")
	case *ssa.Panic:
		panic("not implemented")
	case *ssa.Range:
		panic("not implemented")
	case *ssa.Return:
		// Get the return type of this function
		returnType := llvm.GetReturnType(c.currentFunction(ctx).llvmType)

		// Void types do not need to return anything
		if llvm.GetTypeKind(returnType) != llvm.VoidTypeKind {
			var returnValue llvm.LLVMValueRef

			// Get the return values
			returnValues, err := c.createValues(ctx, instr.Results)
			if err != nil {
				return nil, err
			}

			// Return a tuple if there should be more than one return value.
			// Otherwise, just return the single value.
			if len(instr.Results) > 1 {
				for i, v := range returnValues {
					// Populate the return struct
					returnValue = llvm.BuildInsertValue(c.builder, llvm.GetUndef(returnType), v, uint(i), "")
				}
			} else {
				// Return the single value
				returnValue = returnValues[0]
			}

			// Create the return
			result = llvm.BuildRet(c.builder, returnValue)
		} else {
			// Return nothing
			result = llvm.BuildRetVoid(c.builder)
		}
	case *ssa.RunDefers:
		panic("not implemented")
	case *ssa.Select:
		panic("not implemented")
	case *ssa.Send:
		panic("not implemented")
	case *ssa.Store:
		var err error
		var addr, value llvm.LLVMValueRef
		addr, err = c.createExpression(ctx, instr.Addr)
		if err != nil {
			return nil, err
		}

		value, err = c.createExpression(ctx, instr.Val)
		if err != nil {
			return nil, err
		}

		result = llvm.BuildStore(c.builder, value, addr)

		if variable, ok := c.variables[addr]; ok && c.options.GenerateDebugInfo {
			// Attach debug information
			llvm.DIBuilderInsertDbgValueAtEnd(
				c.dibuilder,
				variable.value,
				variable.dbg,
				llvm.DIBuilderCreateExpression(c.dibuilder, nil),
				c.currentLocation(ctx).line,
				c.currentBlock(ctx))
		}
	case *ssa.DebugRef:
		// Do nothing for now. This statement will be used to insert debug info later
		//panic("not implemented")
	default:
		if value, ok := instr.(ssa.Value); ok {
			// The instruction is an expression. Create the respective instruction.
			result, err = c.createExpression(ctx, value)
			if err != nil {
				return nil, err
			}
		} else {
			panic("encountered unknown instruction")
		}
	}

	return
}

func (c *Compiler) createExpression(ctx context.Context, expr ssa.Value) (value llvm.LLVMValueRef, err error) {
	c.printf(Debug, "Processing expression %T: %s\n", expr, expr.String())

	// Check if this value was cached
	if value, ok := c.values[expr]; ok {
		// This value might be a variable. Get the value from the variable's (alloca) address
		//if _, ok := c.variables[value]; ok {
		//	// Ignore *ssa.Alloc since these should be pointers directly
		//	if _, ok := expr.(*ssa.Alloc); !ok {
		//		value = c.getVariableValue(value)
		//		c.println(Debug, "Loading value from variable")
		//		return value, nil
		//	}
		//}

		// Return the cached value
		c.println(Debug, "returning cached value")
		return value, nil
	}

	// Evaluate the expression
	switch expr := expr.(type) {
	case *ssa.Alloc:
		// Get the type of which memory will be allocated for
		typ := c.createType(ctx, expr.Type().Underlying().(*types.Pointer).Elem())

		// NOTE: Some stack allocations will be moved to the heap later if they
		//       are too big for the stack.
		// Get the size of the type
		size := llvm.StoreSizeOfType(c.options.Target.dataLayout, typ.valueType)

		if expr.Heap {
			// Heap allocations will store the address of the runtime allocation in
			// the alloca. Allocate space for this pointer on the stack.
			//typ = c.createType(ctx, expr.Type().Underlying())

			// Create the alloca to hold the address on the stack
			value = llvm.BuildAlloca(c.builder, c.ptrType.valueType, expr.Comment)

			// Create the runtime call to allocate some memory on the heap
			addr, err := c.createRuntimeCall(ctx, "alloc", []llvm.LLVMValueRef{llvm.ConstInt(c.uintptrType.valueType, size, false)})
			if err != nil {
				return nil, err
			}

			// Store the address at the alloc
			llvm.BuildStore(c.builder, addr, value)
		} else {
			// Create an alloca to hold the value on the stack
			value = llvm.BuildAlloca(c.builder, typ.valueType, expr.Comment)

			// Zero-initialize the stack variable
			if size > 0 {
				llvm.BuildStore(c.builder, llvm.ConstNull(typ.valueType), value)
			}
		}

		// Finally create a variable to hold debug information about this alloca
		//value = c.createVariable(ctx, expr.Name(), value, typ)
	case *ssa.BinOp:
		value, err = c.createBinOp(ctx, expr)
	case *ssa.Call:
		switch callExpr := expr.Common().Value.(type) {
		case *ssa.Builtin:
			value, err = c.createBuiltinCall(ctx, callExpr, expr.Call.Args)
		case *ssa.Function:
			value, err = c.createFunctionCall(ctx, callExpr, expr.Call.Args)
		case *ssa.MakeClosure:
			// a *MakeClosure, indicating an immediately applied function
			// literal with free variables.
			panic("not implemented")
		default:
			// any other value, indicating a dynamically dispatched function
			// call.
			panic("not implemented")
		}
	case *ssa.Const:
		constType := c.createType(ctx, expr.Type())
		if expr.Value == nil {
			value = llvm.ConstNull(constType.valueType)
		} else {
			switch expr.Value.Kind() {
			case constant.Bool:
				if constant.BoolVal(expr.Value) {
					value = llvm.ConstInt(constType.valueType, 1, false)
				} else {
					value = llvm.ConstInt(constType.valueType, 0, false)
				}
			case constant.String:
				strValue := constant.StringVal(expr.Value)
				/*
					// Create the constant string value as a global
					elementType := llvm.Int8TypeInContext(c.currentContext(ctx))
					strConstGlobal := llvm.AddGlobal(c.module, llvm.ArrayType2(elementType, uint64(len(strValue))), "const_string")
					strConstValue := llvm.ConstStringInContext(c.currentContext(ctx), strValue, uint(len(strValue)), false)
					llvm.SetInitializer(strConstGlobal, strConstValue)
					llvm.SetUnnamedAddr(strConstGlobal, llvm.GlobalUnnamedAddr != 0)

					// Create the constant struct value representing strings
					typ := c.createType(ctx, expr.Type())
					value = llvm.ConstNamedStruct(typ.valueType, []llvm.LLVMValueRef{
						llvm.ConstBitCast(strConstGlobal, c.uintptrType.valueType),
						llvm.ConstInt(llvm.Int32Type(), uint64(len(strValue)), false),
					})*/
				value = c.createGlobalString(ctx, strValue)
			case constant.Int:
				constVal, _ := constant.Int64Val(expr.Value)
				value = llvm.ConstInt(constType.valueType, uint64(constVal), false)
			case constant.Float:
				constVal, _ := constant.Float64Val(expr.Value)
				value = llvm.ConstReal(constType.valueType, constVal)
			case constant.Complex:
				panic("not implemented")
			default:
				panic("unknown default value")
			}
		}
	case *ssa.Parameter:
		if fn, ok := c.functions[expr.Parent().Signature]; ok {
			// Locate the parameter in the function
			for i, param := range expr.Parent().Params {
				if param == expr {
					value = llvm.GetParam(fn.value, uint(i))
				}
			}
		} else {
			panic("function does not exist")
		}

		// All parameters should be allocated on the stack.
		/*value = llvm.BuildAlloca(c.builder, typ.valueType, expr.Name())

		// Finally create a variable to hold debug information about this alloca
		//value = c.createVariable(ctx, expr.Name(), alloca, typ)*/
	case *ssa.ChangeInterface:
		panic("not implemented")
	case *ssa.ChangeType:
		panic("not implemented")
	case *ssa.Extract:
		// Get the return struct (tuple)
		structValue, err := c.createExpression(ctx, expr.Tuple)
		if err != nil {
			return nil, err
		}

		// Get the address of the field within the return struct (tuple)
		fieldType, addr := c.structFieldAddress(structValue, expr.Index)

		// Load the value at the address
		value = llvm.BuildLoad2(c.builder, fieldType, addr, "")
	case *ssa.Field:
		structValue, err := c.createExpression(ctx, expr.X)
		if err != nil {
			return nil, err
		}

		// Get the address of the field within the struct
		fieldType, addr := c.structFieldAddress(structValue, expr.Field)

		// Load the value at the address
		value = llvm.BuildLoad2(c.builder, fieldType, addr, "")
	case *ssa.FieldAddr:
		structValue, err := c.createExpression(ctx, expr.X)
		if err != nil {
			return nil, err
		}

		//Return the address
		_, value = c.structFieldAddress(structValue, expr.Field)
	case *ssa.Global:
		// Create a global value
		globalType := c.createType(ctx, expr.Type())
		value = llvm.AddGlobal(c.module, globalType.valueType, fmt.Sprintf("%s.%s", expr.Pkg.Pkg.Path(), expr.Name()))
		// NOTE: This global receives its value later from the package's Init() function. The partial evaluator should
		//       actually be what gives this global its final value instead.
	case *ssa.Index:
		arrayValue, err := c.createExpression(ctx, expr.X)
		if err != nil {
			return nil, err
		}

		indexValue, err := c.createExpression(ctx, expr.Index)
		if err != nil {
			return nil, err
		}

		elementType := llvm.GetElementType(llvm.TypeOf(arrayValue))

		// Get the address of the element at the index within the array
		addr := c.arrayElementAddress(arrayValue, elementType, indexValue)

		// Load the value at the address
		value = llvm.BuildLoad2(c.builder, elementType, addr, "")
	case *ssa.IndexAddr:
		arrayValue, err := c.createExpression(ctx, expr.X)
		if err != nil {
			return nil, err
		}

		indexValue, err := c.createExpression(ctx, expr.Index)
		if err != nil {
			return nil, err
		}

		// The container can be a slice or an array
		if sliceType, ok := expr.X.Type().Underlying().(*types.Slice); ok {
			// Get the element type of the slice
			elementType := c.createType(ctx, sliceType.Elem())

			// Get the element size of the slice
			elementSize := llvm.StoreSizeOfType(c.options.Target.dataLayout, elementType.valueType)

			// Create a runtime call to retrieve the address of the element at index I
			value, err = c.createRuntimeCall(ctx, "sliceIndex", []llvm.LLVMValueRef{
				arrayValue,
				indexValue,
				llvm.ConstInt(llvm.Int64Type(), elementSize, false),
			})
			if err != nil {
				return nil, err
			}
		} else {
			var arrayType llvm.LLVMTypeRef
			if ptrType, ok := expr.X.Type().Underlying().(*types.Pointer); ok {
				// Get the expected array type, so we can cast the pointer
				// value from below to this type.
				arrayType = c.createType(ctx, ptrType.Elem()).valueType

				println("ptr->ptr:", llvm.PrintTypeToString(arrayType))

				// Load the base address of the array from the alloca
				arrayValue = llvm.BuildLoad2(c.builder, c.ptrType.valueType, arrayValue, "")

				// Bitcast the resulting value to a pointer of the array type
				arrayValue = llvm.BuildBitCast(c.builder, arrayValue, llvm.PointerType(arrayType, 0), "")
			} else {
				arrayType = llvm.TypeOf(arrayValue)
				println("ptr?:", llvm.PrintTypeToString(arrayType))
			}

			// Get the address of the element at the index within the array
			value = c.arrayElementAddress(arrayValue, llvm.GetElementType(arrayType), indexValue)
		}
	case *ssa.Lookup:
		panic("not implemented")
	case *ssa.MakeChan:
		panic("not implemented")
	case *ssa.MakeClosure:
		panic("not implemented")
	case *ssa.MakeInterface:
		value, err = c.makeInterface(ctx, expr.X)
		if err != nil {
			return nil, err
		}
	case *ssa.MakeSlice:
		panic("not implemented")
	case *ssa.MultiConvert:
		panic("not implemented")
	case *ssa.Next:
		panic("not implemented")
	case *ssa.Phi:
		phiType := c.createType(ctx, expr.Type())

		// Build the Phi node operator
		value = llvm.BuildPhi(c.builder, phiType.valueType, "")

		// Create a variable for this Phi node
		//value = c.createVariable(ctx, expr.Comment, phiValue, phiType)

		// Cache the PHI value now to prevent a stack overflow in the call to createValues below
		if _, ok := c.values[expr]; ok {
			panic("PHI node value already generated")
		}
		c.values[expr] = value

		// Create values for each of the Phi node's edges
		edgeValues, err := c.createValues(ctx, expr.Edges)
		if err != nil {
			return nil, err
		}

		// Get the blocks each edge value belongs to
		// NOTE: Edges[i] is value for Block().Preds[i]
		var blocks []llvm.LLVMBasicBlockRef
		for i, _ := range expr.Edges {
			block := c.blocks[expr.Block().Preds[i]]
			blocks = append(blocks, block)
		}

		// Add the edges
		llvm.AddIncoming(value, edgeValues, blocks)
	case *ssa.Slice:
		var array, low, high, max llvm.LLVMValueRef
		if expr.Low != nil {
			low, err = c.createExpression(ctx, expr.Low)
			if err != nil {
				return nil, err
			}
		}

		if expr.High != nil {
			high, err = c.createExpression(ctx, expr.High)
			if err != nil {
				return nil, err
			}
		}

		if expr.Max != nil {
			max, err = c.createExpression(ctx, expr.Max)
			if err != nil {
				return nil, err
			}
		}

		array, err = c.createExpression(ctx, expr.X)
		if err != nil {
			return nil, err
		}

		var elementType llvm.LLVMTypeRef
		numElements := uint64(0)

		switch t := expr.X.Type().Underlying().(type) {
		case *types.Slice:
			elementType = c.createType(ctx, t.Elem()).valueType
		case *types.Basic:
			elementType = llvm.Int8Type()
		case *types.Pointer:
			if tt, ok := t.Elem().Underlying().(*types.Array); ok {
				elementType = c.createType(ctx, tt.Elem()).valueType
				numElements = uint64(tt.Len())
			} else {
				panic("invalid pointer type")
			}
		}

		value = c.createSlice(ctx, array, elementType, numElements, low, high, max)
	case *ssa.SliceToArrayPointer:
		panic("not implemented")
	case *ssa.TypeAssert:
		x, err := c.createExpression(ctx, expr.X)
		if err != nil {
			return nil, err
		}

		newType := c.createType(ctx, expr.AssertedType)

		// The value needs to be passed by address. Create an alloca for it and store the value
		xAlloca := llvm.BuildAlloca(c.builder, llvm.TypeOf(x), "")
		llvm.BuildStore(c.builder, x, xAlloca)

		// create the runtime call
		result, err := c.createRuntimeCall(ctx, "typeAssert", []llvm.LLVMValueRef{
			xAlloca,
			c.createTypeDescriptor(ctx, c.createType(ctx, expr.X.Type())),
			c.createTypeDescriptor(ctx, newType),
		})
		if err != nil {
			return nil, err
		}

		// Extract values from return
		// Create a pointer to the struct
		alloca := llvm.BuildAlloca(c.builder, llvm.TypeOf(result), "")
		llvm.BuildStore(c.builder, result, alloca)
		_, newObj := c.structFieldAddress(alloca, 0)
		_, isOk := c.structFieldAddress(alloca, 1)

		// TODO: There definitely more involved than this. Gonna try to
		//       implement the semantics in Go code rather than hardcode it
		//       here. I should be able to load the resulting object from
		//       the address obtained from the runtime call.
		obj := llvm.BuildLoad2(c.builder, newType.valueType, newObj, "")

		if expr.CommaOk {
			// Return the obj and the status
			value = llvm.ConstStruct([]llvm.LLVMValueRef{
				llvm.GetUndef(newType.valueType),
				llvm.GetUndef(llvm.Int1Type()),
			}, false)
			value = llvm.BuildInsertValue(c.builder, value, obj, uint(0), "")
			value = llvm.BuildInsertValue(c.builder, value, isOk, uint(1), "")
		} else {

			value = obj
		}
	case *ssa.UnOp:
		value, err = c.createUpOp(ctx, expr)
	}

	if value == nil {
		panic("nil value")
	}

	// Cache the value
	// NOTE: PHIs need to cache their value earlier to prevent a stack overflow
	if _, ok := expr.(*ssa.Phi); !ok {
		c.values[expr] = value
	}

	return value, nil
	//return c.getVariableValue(value), nil
}

func (c *Compiler) createFunctionCall(ctx context.Context, callee *ssa.Function, args []ssa.Value) (llvm.LLVMValueRef, error) {
	// Get the info about the callee
	fn, ok := c.functions[callee.Signature]
	if !ok {
		panic("call to function that does not exist")
	}

	// Create the argument values
	values, err := c.createValues(ctx, args)
	if err != nil {
		return nil, err
	}

	// Create and return the value of the call
	return llvm.BuildCall2(c.builder, fn.llvmType, fn.value, values, ""), nil
}

func (c *Compiler) createRuntimeCall(ctx context.Context, name string, args []llvm.LLVMValueRef) (llvm.LLVMValueRef, error) {
	c.println(Debug, "Creating runtime call:", name)
	// Get the runtime function
	fn := llvm.GetNamedFunction(c.module, "runtime."+name)
	if fn == nil {
		panic("runtime does not implement " + name)
	}

	var fnType llvm.LLVMTypeRef

	// Locate the information about this function
	for _, info := range c.functions {
		if info.value == fn {
			fnType = info.llvmType
			break
		}
	}

	if fnType == nil {
		panic("function not cached?")
	}

	// Process each argument value to make sure they are the correct types as
	// defined by the function.
	for i, t := range llvm.GetParamTypes(fnType) {
		argType := llvm.TypeOf(args[i])
		if !llvm.TypeIsEqual(argType, t) {
			panic("argument type does not match")
		}
	}

	// Create and return the value of the call
	return llvm.BuildCall2(c.builder, fnType, fn, args, ""), nil
}

func (c *Compiler) createValues(ctx context.Context, input []ssa.Value) ([]llvm.LLVMValueRef, error) {
	var output []llvm.LLVMValueRef
	for _, in := range input {
		// Evaluate the argument
		value, err := c.createExpression(ctx, in)
		if err != nil {
			return nil, err
		}

		// Append to the args list that will be passed to the function
		output = append(output, value)
	}
	return output, nil
}

func (c *Compiler) createVariable(ctx context.Context, name string, value llvm.LLVMValueRef, valueType Type) llvm.LLVMValueRef {
	// Check the value type. All variables should have an alloca associated with it. If the type is not an alloca then
	// create one in the entry block and store the value into it.
	var alloca llvm.LLVMValueRef
	if alloca = llvm.IsAAllocaInst(value); alloca == nil {
		alloca = llvm.BuildAlloca(c.builder, llvm.TypeOf(value), name)

		// Store the value into the alloca
		llvm.BuildStore(c.builder, value, alloca)
	}

	// Create and associate the variable with the value of the alloca
	variable := Variable{
		value: alloca,
	}

	if c.options.GenerateDebugInfo {
		// Create the debug information about the variable
		variable.dbg = llvm.DIBuilderCreateAutoVariable(c.dibuilder,
			c.currentScope(ctx),
			name,
			c.currentLocation(ctx).file,
			uint(c.currentLocation(ctx).position.Line),
			valueType.debugType,
			true, 0, 0)

		// Add debug info about the declaration
		llvm.DIBuilderInsertDeclareAtEnd(
			c.dibuilder,
			variable.value,
			variable.dbg,
			llvm.DIBuilderCreateExpression(c.dibuilder, nil),
			c.currentLocation(ctx).line,
			c.currentBlock(ctx))
	}

	// Store the variable
	c.variables[alloca] = &variable

	// Return the alloca value
	return alloca
}

func (c *Compiler) structFieldAddress(value llvm.LLVMValueRef, index int) (llvm.LLVMTypeRef, llvm.LLVMValueRef) {
	// The value must be a pointer to a struct
	valueType := llvm.TypeOf(value)
	kind := llvm.GetTypeKind(valueType)
	if kind == llvm.PointerTypeKind {
		if llvm.IsAAllocaInst(value) != nil {
			// Get the allocated type
			valueType = llvm.GetAllocatedType(value)
			kind = llvm.GetTypeKind(valueType)
			if kind != llvm.StructTypeKind {
				panic("attempting to extract field value from non pointer-to-struct value")
			}
		}
	}

	// Get the type of the field
	fieldType := llvm.StructGetTypeAtIndex(valueType, uint(index))

	// Create a GEP to get the  address of the field in the struct
	return fieldType, llvm.BuildStructGEP2(c.builder, valueType, value, uint(index), "")
}

func (c *Compiler) arrayElementAddress(value llvm.LLVMValueRef, elementType llvm.LLVMTypeRef, index llvm.LLVMValueRef) llvm.LLVMValueRef {
	// Must be a pointer to an array
	//valueType := llvm.TypeOf(value)
	//kind := llvm.GetTypeKind(valueType)
	//if kind != llvm.ArrayTypeKind {
	//	panic("attempting to extract value from non-array value")
	//}

	// Create an array of values. The indices in the array are used to access
	// inner and outer structures. In the case of an array the outer structure
	// is the array so 0 is passed first and then the index of the inner
	// element next.
	indices := []llvm.LLVMValueRef{
		//llvm.ConstInt(llvm.Int32Type(), 0, false),
		index,
	}

	// Get the address of the value at the index in the array
	println(llvm.PrintTypeToString(elementType))
	return llvm.BuildGEP2(c.builder, elementType, value, indices, "")
}

func (c *Compiler) getVariableValue(value llvm.LLVMValueRef) llvm.LLVMValueRef {
	valueType := llvm.TypeOf(value)
	kind := llvm.GetTypeKind(valueType)
	if kind == llvm.PointerTypeKind {
		if llvm.IsAAllocaInst(value) != nil {
			// Get the allocated type
			valueType = llvm.GetAllocatedType(value)

			// Load the value
			value = llvm.BuildLoad2(c.builder, valueType, value, "")
		}
	}

	return value
}
