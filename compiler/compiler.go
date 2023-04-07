package compiler

import (
	"context"
	"go/token"
	"go/types"
	"golang.org/x/tools/go/ssa"
	"omibyte.io/sigo/llvm"
	"path/filepath"
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
	valueType llvm.LLVMTypeRef
	debugType llvm.LLVMMetadataRef
}

type Location struct {
	file     llvm.LLVMMetadataRef
	line     llvm.LLVMMetadataRef
	position token.Position
}

type Compiler struct {
	module      llvm.LLVMModuleRef
	builder     llvm.LLVMBuilderRef
	dibuilder   llvm.LLVMDIBuilderRef
	functions   map[*types.Signature]*Function
	types       map[types.Type]Type
	values      map[ssa.Value]llvm.LLVMValueRef
	variables   map[llvm.LLVMValueRef]*Variable
	uintptrType Type
	target      *Target

	GenerateDebugInfo bool
}

func NewCompiler(target *Target) *Compiler {
	// Create the LLVM context
	ctx := llvm.ContextCreate()

	// Create the module
	module := llvm.ModuleCreateWithNameInContext("main", ctx)

	// Set the data layout for this module from the target
	llvm.SetDataLayout(module, llvm.CopyStringRepOfTargetData(target.dataLayout))

	// Create the instruction builder
	builder := llvm.CreateBuilderInContext(ctx)

	// Create the DIBuilder
	dibuilder := llvm.CreateDIBuilder(module)

	// Create the uintptr type based on the machine's pointer width
	uintptrType := Type{
		valueType: llvm.IntPtrTypeInContext(ctx, target.dataLayout),
	}

	// Create and return the compiler
	return &Compiler{
		module:            module,
		builder:           builder,
		dibuilder:         dibuilder,
		functions:         map[*types.Signature]*Function{},
		types:             map[types.Type]Type{},
		values:            map[ssa.Value]llvm.LLVMValueRef{},
		variables:         map[llvm.LLVMValueRef]*Variable{},
		uintptrType:       uintptrType,
		target:            target,
		GenerateDebugInfo: true,
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

	if c.GenerateDebugInfo {
		// Create the debug info for the pointer type
		c.uintptrType.debugType = llvm.DIBuilderCreateBasicType(
			c.dibuilder, "uintptr", uint64(llvm.PointerSize(c.target.dataLayout)), DW_ATE_unsigned, 0)
	}

	// Compile each member of the package
	for _, member := range pkg.Members {
		switch member := member.(type) {
		case *ssa.NamedConst:
			panic("Not implemented")
		case *ssa.Global:
			panic("Not implemented")
		case *ssa.Type:
			// Create the type
			c.createType(ctx, member.Type())
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
		if err := c.createFunctionBlocks(ctx, fn); err != nil {
			return err
		}
	}

	return nil
}

func (c *Compiler) createFunction(ctx context.Context, fn *ssa.Function) (*Function, error) {
	var returnValueTypes []llvm.LLVMTypeRef
	var argValueTypes []llvm.LLVMTypeRef
	var argDiTypes []llvm.LLVMMetadataRef

	var returnType llvm.LLVMTypeRef
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
	fnValue := llvm.AddFunction(c.module, fn.Name(), fnType)

	result := Function{
		value:    fnValue,
		llvmType: fnType,
		def:      fn,
	}

	// Create the file information for this function
	if c.GenerateDebugInfo {
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
		panic("attempting to generate a function more than once")
	}

	// Set the current function type in the context
	ctx = context.WithValue(ctx, currentFnTypeKey{}, fn)

	// Create the entry block for this function
	entryBlock := llvm.AppendBasicBlockInContext(c.currentContext(ctx), fn.value, fn.def.Name()+".entry")

	// Set the current entry block in the context
	ctx = context.WithValue(ctx, entryBlockKey{}, entryBlock)

	var file *token.File
	var scope llvm.LLVMMetadataRef

	if c.GenerateDebugInfo {
		// Get the file information for this function
		file = fn.def.Prog.Fset.File(fn.def.Pos())

		// Create the scope debug information for this function
		scope = llvm.GetSubprogram(fn.value)
		ctx = context.WithValue(ctx, currentScopeKey{}, scope)
	}

	// Create the function's blocks
	for _, block := range fn.def.Blocks {
		var insertionBlock llvm.LLVMBasicBlockRef
		// Is this the entry block?
		if block.Comment == "entry" {
			// Proceed with the entry block
			insertionBlock = entryBlock
		} else {
			// Create a new block
			insertionBlock = llvm.AppendBasicBlockInContext(c.currentContext(ctx), fn.value, fn.def.Name()+"."+block.Comment)
		}

		// All further instructions should go into this block
		llvm.PositionBuilderAtEnd(c.builder, insertionBlock)

		// Set the current block in the context
		ctx = context.WithValue(ctx, currentBlockKey{}, insertionBlock)

		// Create each instruction in the block
		for _, instr := range block.Instrs {
			if c.GenerateDebugInfo {
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
	}

	// Mark this function as compiled
	fn.compiled = true

	return nil
}

func (c *Compiler) createType(ctx context.Context, typ types.Type) Type {
	// Check if this type was already created
	if result, ok := c.types[typ]; ok {
		return result
	}

	// Is this a named type?
	name := ""
	if named, ok := typ.(*types.Named); ok {
		// Extract the name
		name = named.Obj().Name()

		// Create the underlying type
		typ = named.Underlying()
	}

	var result Type

	// Create the equivalent LLVM type
	switch typ := typ.(type) {
	case *types.Basic:
		switch typ.Kind() {
		case types.Uint8:
			result.valueType = llvm.Int8TypeInContext(c.currentContext(ctx))
			if c.GenerateDebugInfo {
				result.debugType = llvm.DIBuilderCreateBasicType(c.dibuilder, typ.Name(), 1, DW_ATE_unsigned, 0)
			}
		case types.Int8:
			result.valueType = llvm.Int8TypeInContext(c.currentContext(ctx))
			if c.GenerateDebugInfo {
				result.debugType = llvm.DIBuilderCreateBasicType(c.dibuilder, typ.Name(), 1, DW_ATE_signed, 0)
			}
		case types.Uint16:
			result.valueType = llvm.Int16TypeInContext(c.currentContext(ctx))
			if c.GenerateDebugInfo {
				result.debugType = llvm.DIBuilderCreateBasicType(c.dibuilder, typ.Name(), 2, DW_ATE_unsigned, 0)
			}
		case types.Int16:
			result.valueType = llvm.Int16TypeInContext(c.currentContext(ctx))
			if c.GenerateDebugInfo {
				result.debugType = llvm.DIBuilderCreateBasicType(c.dibuilder, typ.Name(), 2, DW_ATE_signed, 0)
			}
		case types.Int, types.Int32:
			result.valueType = llvm.Int32TypeInContext(c.currentContext(ctx))
			if c.GenerateDebugInfo {
				result.debugType = llvm.DIBuilderCreateBasicType(c.dibuilder, typ.Name(), 4, DW_ATE_unsigned, 0)
			}
		case types.Uint, types.Uint32:
			result.valueType = llvm.Int32TypeInContext(c.currentContext(ctx))
			if c.GenerateDebugInfo {
				result.debugType = llvm.DIBuilderCreateBasicType(c.dibuilder, typ.Name(), 4, DW_ATE_signed, 0)
			}
		case types.Uint64:
			result.valueType = llvm.Int64TypeInContext(c.currentContext(ctx))
			if c.GenerateDebugInfo {
				result.debugType = llvm.DIBuilderCreateBasicType(c.dibuilder, typ.Name(), 8, DW_ATE_unsigned, 0)
			}
		case types.Int64:
			result.valueType = llvm.Int64TypeInContext(c.currentContext(ctx))
			if c.GenerateDebugInfo {
				result.debugType = llvm.DIBuilderCreateBasicType(c.dibuilder, typ.Name(), 8, DW_ATE_signed, 0)
			}
		case types.Uintptr:
			return c.uintptrType
		case types.UnsafePointer:
			result = c.uintptrType
		case types.Float32:
			result.valueType = llvm.FloatTypeInContext(c.currentContext(ctx))
			if c.GenerateDebugInfo {
				result.debugType = llvm.DIBuilderCreateBasicType(c.dibuilder, typ.Name(), 4, DW_ATE_float, 0)
			}
		case types.Float64:
			result.valueType = llvm.DoubleTypeInContext(c.currentContext(ctx))
			if c.GenerateDebugInfo {
				result.debugType = llvm.DIBuilderCreateBasicType(c.dibuilder, typ.Name(), 8, DW_ATE_float, 0)
			}
		case types.Complex64:
			panic("Not implemented")
		case types.Complex128:
			panic("Not implemented")
		case types.String:
			result.valueType = llvm.StructCreateNamed(c.currentContext(ctx), "string")
			structMembers := []llvm.LLVMTypeRef{
				c.uintptrType.valueType,                        // Ptr
				llvm.Int32TypeInContext(c.currentContext(ctx)), // Len
			}
			llvm.StructSetBody(result.valueType, structMembers, false)

			if c.GenerateDebugInfo {
				// Create the debug information for the underlying struct
				result.debugType = llvm.DIBuilderCreateStructType(
					c.dibuilder,
					c.currentScope(ctx),
					"string",
					nil,
					0,
					4, // Alignment - TODO: Get this information from the target information
					0, 0, nil, nil, 0, nil, "")
				llvm.DIBuilderCreateMemberType(
					c.dibuilder,
					result.debugType,
					"array",
					nil,
					0,
					4,
					4, // Alignment - TODO: Get this information from the target information
					0, // Offset
					0, // Flags
					c.uintptrType.debugType)
				llvm.DIBuilderCreateMemberType(
					c.dibuilder,
					result.debugType,
					"len",
					nil,
					0,
					4,
					4,  // Alignment - TODO: Get this information from the target information
					32, // Offset - TODO: Use the value above to calculate the offset within the struct
					0,  // Flags
					llvm.DIBuilderCreateBasicType(c.dibuilder, "int", 4, DW_ATE_unsigned, 0))
			}
		default:
			panic("encountered unknown basic type")
		}
	case *types.Struct:
		if c.GenerateDebugInfo {
			result.debugType = llvm.DIBuilderCreateStructType(
				c.dibuilder,
				c.currentScope(ctx),
				name,
				nil,
				0,
				4, // Alignment - TODO: Get this information from the target information
				0, 0, nil, nil, 0, nil, "")
		}

		var structMembers []llvm.LLVMTypeRef
		offset := uint64(0)
		for i := 0; i < typ.NumFields(); i++ {
			field := typ.Field(i)
			fieldType := c.createType(ctx, field.Type())
			structMembers = append(structMembers, fieldType.valueType)

			if c.GenerateDebugInfo {
				llvm.DIBuilderCreateMemberType(
					c.dibuilder,
					result.debugType,
					field.Name(),
					nil,
					0,
					4,
					4,      // Alignment - TODO: Get this information from the target information
					offset, // Offset
					0,      // Flags
					fieldType.debugType)

				offset += llvm.DITypeGetSizeInBits(fieldType.debugType)
			}
		}
		result.valueType = llvm.StructCreateNamed(c.currentContext(ctx), name)
		llvm.StructSetBody(result.valueType, structMembers, false)
	case *types.Map:
		panic("Not implemented")
	case *types.Slice:
		structMembers := []llvm.LLVMTypeRef{
			c.uintptrType.valueType,                        // Ptr
			llvm.Int32TypeInContext(c.currentContext(ctx)), // Len
			llvm.Int32TypeInContext(c.currentContext(ctx)), // Cap
		}
		result.valueType = llvm.StructCreateNamed(c.currentContext(ctx), name)
		llvm.StructSetBody(result.valueType, structMembers, false)

		if c.GenerateDebugInfo {
			// Create the debug information for the underlying struct
			result.debugType = llvm.DIBuilderCreateStructType(
				c.dibuilder,
				c.currentScope(ctx),
				name,
				nil,
				0,
				4, // Alignment - TODO: Get this information from the target information
				0, 0, nil, nil, 0, nil, "")
			llvm.DIBuilderCreateMemberType(
				c.dibuilder,
				result.debugType,
				"array",
				nil,
				0,
				4,
				4, // Alignment - TODO: Get this information from the target information
				0, // Offset
				0, // Flags
				c.uintptrType.debugType)
			llvm.DIBuilderCreateMemberType(
				c.dibuilder,
				result.debugType,
				"len",
				nil,
				0,
				4,
				4,  // Alignment - TODO: Get this information from the target information
				32, // Offset - TODO: Use the value above to calculate the offset within the struct
				0,  // Flags
				llvm.DIBuilderCreateBasicType(c.dibuilder, "int", 4, DW_ATE_unsigned, 0))
			llvm.DIBuilderCreateMemberType(
				c.dibuilder,
				result.debugType,
				"cap",
				nil,
				0,
				4,
				4,  // Alignment - TODO: Get this information from the target information
				64, // Offset - TODO: Use the value above to calculate the offset within the struct
				0,  // Flags
				llvm.DIBuilderCreateBasicType(c.dibuilder, "int", 4, DW_ATE_unsigned, 0))
		}
	case *types.Array:
		elementType := c.createType(ctx, typ.Elem())
		result.valueType = llvm.ArrayType2(elementType.valueType, uint64(typ.Len()))
		if c.GenerateDebugInfo {
			result.debugType = llvm.DIBuilderCreateArrayType(
				c.dibuilder,
				uint64(typ.Len()),
				4, // TODO: Get this information for the target machine descriptor
				elementType.debugType,
				nil)
		}
	case *types.Interface:
		structMembers := []llvm.LLVMTypeRef{
			c.uintptrType.valueType, // Ptr
			c.uintptrType.valueType, // Vtable
		}
		result.valueType = llvm.StructCreateNamed(c.currentContext(ctx), name)
		llvm.StructSetBody(result.valueType, structMembers, false)

		if c.GenerateDebugInfo {
			// Create the debug information for the underlying struct
			result.debugType = llvm.DIBuilderCreateStructType(
				c.dibuilder,
				c.currentScope(ctx),
				name,
				nil,
				0,
				4, // Alignment - TODO: Get this information from the target information
				0, 0, nil, nil, 0, nil, "")

			llvm.DIBuilderCreateMemberType(
				c.dibuilder,
				result.debugType,
				"ptr",
				nil,
				0,
				4,
				4,  // Alignment - TODO: Get this information from the target information
				64, // Offset - TODO: Use the value above to calculate the offset within the struct
				0,  // Flags
				c.uintptrType.debugType)

			llvm.DIBuilderCreateMemberType(
				c.dibuilder,
				result.debugType,
				"vtable",
				nil,
				0,
				4,
				4,  // Alignment - TODO: Get this information from the target information
				64, // Offset - TODO: Use the value above to calculate the offset within the struct
				0,  // Flags
				c.uintptrType.debugType)
		}
	case *types.Pointer:
		elementType := c.createType(ctx, typ.Elem())
		result.valueType = llvm.PointerType(elementType.valueType, 0)
		if c.GenerateDebugInfo {
			result.debugType = llvm.DIBuilderCreateObjectPointerType(c.dibuilder, elementType.debugType)
		}
	default:
		panic("encountered unknown type")
	}

	// A known type should be created under all circumstances
	if result.valueType == nil {
		panic("no type was created")
	}

	// Cache the type for faster lookup later
	c.types[typ] = result

	// Return the type
	return result
}

func (c *Compiler) createInstruction(ctx context.Context, instr ssa.Instruction) (result llvm.LLVMValueRef, err error) {
	// Create the specific instruction
	switch instr := instr.(type) {
	case *ssa.Defer:
		panic("not implemented")
	case *ssa.Go:
		panic("not implemented")
	case *ssa.If:
		panic("not implemented")
	case *ssa.Jump:
		panic("not implemented")
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
				// Populate the return struct
				returnValue = llvm.ConstNamedStruct(returnType, returnValues)
			} else {
				// Return undef value
				returnValue = returnValues[0]
			}

			// Create the return
			result = llvm.BuildRet(c.builder, returnValue)
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

		if variable, ok := c.variables[addr]; ok && c.GenerateDebugInfo {
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

			// TODO: Cache the value

		} else {
			panic("encountered unknown instruction")
		}
	}

	return
}

func (c *Compiler) createExpression(ctx context.Context, expr ssa.Value) (value llvm.LLVMValueRef, err error) {
	// Check if this value was cached
	if value, ok := c.values[expr]; ok {
		// Return the cached value
		return value, nil
	}

	// Evaluate the expression
	switch expr := expr.(type) {
	case *ssa.Alloc:
		// Get the type of which memory will be allocated for
		typ := c.createType(ctx, expr.Type())

		// Create an alloca to hold the value on the stack
		alloca := llvm.BuildAlloca(c.builder, typ.valueType, expr.Comment)

		// Heap allocations will store the address of the runtime allocation in
		// the alloca.

		// NOTE: Some stack allocations will be moved to the heap later if they
		//       are too big for the stack.
		if expr.Heap {
			// Get the size of the pointer's element type
			//elementType := c.createType(ctx, expr.Type().Underlying().(*types.Pointer).Elem())
			//size := llvm.ConstInt(llvm.StoreSizeOfType())
			//heapAllocAddr := c.createRuntimeCall(ctx, "alloc")

			panic("not implemented")
		}

		// Finally create a variable to hold debug information about this alloca
		value = c.createVariable(ctx, expr.Name(), alloca, typ)
	case *ssa.BinOp:
		panic("not implemented")
	case *ssa.Call:
		if builtin, ok := expr.Common().Value.(*ssa.Builtin); ok {
			value, err = c.createBuiltinCall(ctx, builtin, expr.Call.Args)
			if err != nil {
				return
			}
		} else {

		}
	case *ssa.Parameter:
		// Get the type of which memory will be allocated for
		typ := c.createType(ctx, expr.Type())

		// All parameters should be allocated on the stack.
		alloca := llvm.BuildAlloca(c.builder, typ.valueType, "test")

		// Finally create a variable to hold debug information about this alloca
		value = c.createVariable(ctx, expr.Name(), alloca, typ)
	case *ssa.ChangeInterface:
		panic("not implemented")
	case *ssa.ChangeType:
		panic("not implemented")
	case *ssa.Extract:
		panic("not implemented")
	case *ssa.Field:
		panic("not implemented")
	case *ssa.FieldAddr:
		// Get the address of the struct
		x, err := c.createExpression(ctx, expr.X)
		if err != nil {
			return nil, err
		}

		// Get the struct type
		typ := c.createType(ctx, expr.Type().(*types.Pointer).Elem())

		// Create a GEP to get the value of the struct field
		value = llvm.BuildStructGEP2(c.builder, typ.valueType, x, uint(expr.Field), "")
	case *ssa.Index:
		panic("not implemented")
	case *ssa.IndexAddr:
		panic("not implemented")
	case *ssa.Lookup:
		panic("not implemented")
	case *ssa.MakeChan:
		panic("not implemented")
	case *ssa.MakeClosure:
		panic("not implemented")
	case *ssa.MakeInterface:
		panic("not implemented")
	case *ssa.MakeSlice:
		panic("not implemented")
	case *ssa.MultiConvert:
		panic("not implemented")
	case *ssa.Next:
		panic("not implemented")
	case *ssa.Phi:
		panic("not implemented")
	case *ssa.Slice:
		panic("not implemented")
	case *ssa.SliceToArrayPointer:
		panic("not implemented")
	case *ssa.TypeAssert:
		panic("not implemented")
	case *ssa.UnOp:
		// Get the value of X
		x, err := c.createExpression(ctx, expr.X)
		if err != nil {
			return nil, err
		}

		// Get the type of X
		xType := llvm.TypeOf(x)

		// Create the respective LLVM operator
		switch expr.Op {
		case token.ARROW:
			// TODO: Channel receive
			panic("Not implemented")
		case token.MUL:
			// Pointer indirection. Create a load operator.
			value = llvm.BuildLoad2(c.builder, xType, x, "")
		case token.NOT:
			// Logical negation
			value = llvm.BuildNot(c.builder, x, "")
		case token.SUB:
			// Negation
			value = llvm.BuildNeg(c.builder, x, "")
		case token.XOR:
			// Bitwise complement
			value = llvm.BuildXor(c.builder, x, llvm.ConstAllOnes(xType), "")
		}
	}

	// Cache the value
	c.values[expr] = value

	return value, nil
}

func (c *Compiler) createBuiltinCall(ctx context.Context, builtin *ssa.Builtin, args []ssa.Value) (value llvm.LLVMValueRef, err error) {
	// Create the proper builtin call based on the method and the type
	// NOTE: It MUST be defined in the runtime used by this application
	switch builtin.Name() {
	case "append":
	case "cap":
	case "close":
	case "complex":
	case "copy":
	case "delete":
	case "imag":
	case "len":
		value, err = c.createSliceLenCall(ctx, builtin, args[0])
	case "make":
	case "new":
	case "panic":
	case "print":
	case "println":
	case "real":
	case "recover":
	default:
		panic("encountered unknown builtin function")
	}

	return
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
	fn := llvm.GetNamedFunction(c.module, "runtime."+name)
	if fn == nil {
		panic("runtime does not implement " + name)
	}

	// Create and return the value of the call
	return llvm.BuildCall2(c.builder, llvm.GetElementType(llvm.TypeOf(fn)), fn, args, ""), nil
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

	if c.GenerateDebugInfo {
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
