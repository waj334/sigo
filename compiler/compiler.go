package compiler

import (
	"context"
	"fmt"
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
	value      llvm.LLVMValueRef
	llvmType   llvm.LLVMTypeRef
	def        *ssa.Function
	diFile     llvm.LLVMMetadataRef
	subprogram llvm.LLVMMetadataRef
	compiled   bool
	hasDefer   bool
	name       string
}

type Type struct {
	valueType  llvm.LLVMTypeRef
	debugType  llvm.LLVMMetadataRef
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
	compileUnit llvm.LLVMMetadataRef
	functions   map[*types.Signature]*Function
	types       map[types.Type]*Type
	descriptors map[types.Type]llvm.LLVMValueRef
	values      map[ssa.Value]Value
	phis        map[*ssa.Phi]Phi
	blocks      map[*ssa.BasicBlock]llvm.LLVMBasicBlockRef

	uintptrType Type
	ptrType     Type
}

var (
	invalidValue Value = Value{}
)

func NewCompiler(options Options) (*Compiler, llvm.LLVMContextRef) {
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

	// Create a compile unit
	cu := llvm.DIBuilderCreateCompileUnit(
		dibuilder,
		llvm.LLVMDWARFSourceLanguage(llvm.DWARFSourceLanguageGo),
		llvm.DIBuilderCreateFile(dibuilder, "<unknown>", ""),
		"SiGo Alpha 0.0.0", // TODO: Use a constant for this
		false,
		"",
		0,
		"",
		llvm.LLVMDWARFEmissionKind(llvm.DWARFEmissionFull),
		0,
		false,
		false,
		"",
		"",
	)

	// Create the uintptr type based on the machine's pointer width
	size := llvm.SizeOfTypeInBits(options.Target.dataLayout,
		llvm.IntPtrTypeInContext(ctx, options.Target.dataLayout))

	uintptrType := Type{
		valueType: llvm.IntPtrTypeInContext(ctx, options.Target.dataLayout),
		debugType: llvm.DIBuilderCreateBasicType(dibuilder, "uint", size, DW_ATE_unsigned, 0),
	}

	ptrType := Type{
		valueType: llvm.PointerType(uintptrType.valueType, 0),
		debugType: llvm.DIBuilderCreateBasicType(dibuilder, "void*", size, DW_ATE_address, 0),
	}

	// Create and return the compiler
	cc := Compiler{
		options:     options,
		module:      module,
		builder:     builder,
		dibuilder:   dibuilder,
		compileUnit: cu,
		functions:   map[*types.Signature]*Function{},
		types:       map[types.Type]*Type{},
		descriptors: map[types.Type]llvm.LLVMValueRef{},
		values:      map[ssa.Value]Value{},
		blocks:      map[*ssa.BasicBlock]llvm.LLVMBasicBlockRef{},
		phis:        map[*ssa.Phi]Phi{},
		uintptrType: uintptrType,
		ptrType:     ptrType,
	}

	// Initialize the type system
	cc.createPrimitiveTypes(ctx)
	cc.createTypeInfoTypes(ctx)

	return &cc, ctx
}

func (c *Compiler) Module() llvm.LLVMModuleRef {
	return c.module
}

func (c *Compiler) Finalize() {
	// Resolve all debug metadata nodes
	llvm.DIBuilderFinalize(c.dibuilder)
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
	if value, ok := ctx.Value(currentLocationKey{}).(Location); ok {
		return value
	}
	return Location{}
}

func (c *Compiler) CompilePackage(ctx context.Context, llvmCtx llvm.LLVMContextRef, pkg *ssa.Package) error {
	// Set the LLVM context to the compiler context
	ctx = context.WithValue(ctx, llvmContextKey{}, llvmCtx)

	// Set the current package in the context
	ctx = context.WithValue(ctx, currentPackageKey{}, pkg)

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
			value, err := c.createExpression(ctx, member)
			if err != nil {
				return err
			}

			value.global = true

			// Create the debug info for the global var (
			value.dbg = llvm.DIBuilderCreateGlobalVariableExpression(
				c.dibuilder,
				c.compileUnit,
				member.Name(),
				member.Object().Id(),
				value.DebugFile(),
				uint(value.Pos().Line),
				c.createDebugType(ctx, member.Type().Underlying().(*types.Pointer).Elem()),
				true,
				llvm.DIBuilderCreateExpression(c.dibuilder, nil),
				nil,
				0)

			llvm.GlobalSetMetadata(value, llvm.GetMDKindID("dbg"), value.dbg)

			// Cache the global
			c.values[member] = value
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
	var returnType llvm.LLVMTypeRef

	isExported := false
	isExternal := false

	// Determine the function name. //go:linkname can override this
	name := types.Id(c.currentPackage(ctx).Pkg, fn.Name())

	// Process any pragmas
	if info, ok := c.options.Symbols[name]; ok {
		if len(info.LinkName) > 0 {
			// Override the generated name
			name = info.LinkName
		}

		// Attempt to find the existing function with the same linkname
		if fnValue := llvm.GetNamedFunction(c.module, name); fnValue != nil {
			// Find the function struct with the matching value
			for _, existingFn := range c.functions {
				if existingFn.value == fnValue {
					// Assume that this function will actually provide the implementation if it has blocks and the
					// existing doesn't. Panic if both have blocks
					if len(fn.Blocks) > 0 {
						if len(existingFn.def.Blocks) == 0 {
							// Override
							existingFn.def = fn

							// TODO: Should override debug information to reflect
							//       this incoming function instead of the
							//       predeclared one.
						} else {
							// TODO: Throw a nice compiler error instead of a panic
							panic("another function has provided the implementation")
						}
					}

					// Return this function directly so that they are "linked".
					return existingFn, nil
				}
			}
		}

		isExported = info.Exported
		isExternal = info.ExternalLinkage
	}

	// Cannot be both exported and external
	if isExported && isExternal {
		panic("function cannot be both external and exported")
	}

	if _, ok := c.functions[fn.Signature]; ok {
		panic("function already compiled")
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
	}

	// Create the function type
	fnType := llvm.FunctionType(returnType, argValueTypes, fn.Signature.Variadic())

	// Add the function to the current module
	fnValue := llvm.AddFunction(c.module, name, fnType)

	if !isExported {
		// Set export status
		llvm.SetVisibility(fnValue, llvm.LLVMVisibility(llvm.HiddenVisibility))
	}

	if !isExternal {
		llvm.SetLinkage(fnValue, llvm.LLVMLinkage(llvm.ExternalLinkage))
	}

	result := Function{
		value:    fnValue,
		llvmType: fnType,
		def:      fn,
		name:     name,
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

	// Get the file information for this function
	file := fn.def.Prog.Fset.File(fn.def.Pos())

	// Some functions, like package initializers, do not have position information
	if file != nil {
		// Extract the file info
		filename := c.options.MapPath(file.Name())
		line := uint(file.Line(fn.def.Pos()))

		// Create the debug information for this function
		var argDiTypes []llvm.LLVMMetadataRef
		for _, arg := range fn.def.Params {
			argDiTypes = append(argDiTypes, c.createDebugType(ctx, arg.Type()))
		}

		fn.diFile = llvm.DIBuilderCreateFile(
			c.dibuilder,
			filepath.Base(filename),
			filepath.Dir(filename))

		subType := llvm.DIBuilderCreateSubroutineType(
			c.dibuilder,
			fn.diFile,
			argDiTypes,
			0)

		fn.subprogram = llvm.DIBuilderCreateFunction(
			c.dibuilder,
			fn.diFile,
			fn.name,
			fn.name,
			fn.diFile,
			line,
			subType,
			true,
			true, 0, llvm.LLVMDIFlags(llvm.DIFlagPrototyped), false)

		// Apply this metadata to the function
		llvm.SetSubprogram(fn.value, fn.subprogram)
		scope := llvm.DIBuilderCreateLexicalBlock(
			c.dibuilder,
			fn.subprogram,
			fn.diFile,
			line,
			0,
		)
		ctx = context.WithValue(ctx, currentScopeKey{}, scope)
	} else {
		// TODO: create fake location information for synthetic functions (initializers, etc...)
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
		for ii, instr := range block.Instrs {
			c.printf(Debug, "Begin instruction #%d\n", ii)
			if fn.subprogram != nil {
				// Get the location information for this instruction
				locationInfo := file.Position(instr.Pos())

				// Create the file debug information
				location := Location{
					file: fn.diFile,
					line: llvm.DIBuilderCreateDebugLocation(
						c.currentContext(ctx),
						uint(locationInfo.Line),
						uint(locationInfo.Column),
						fn.subprogram,
						nil,
					),
					position: locationInfo,
				}

				// Set the current location in the context
				ctx = context.WithValue(ctx, currentLocationKey{}, location)
				llvm.SetCurrentDebugLocation2(c.builder, location.line)
			}

			// Create the instruction
			if err := c.createInstruction(ctx, instr); err != nil {
				return err
			}
			c.printf(Debug, "End instruction #%d\n", ii)
		}
		c.println(Debug, "Done processing block", i)
	}

	// Mark this function as compiled
	fn.compiled = true

	if fn.subprogram != nil {
		// Subprograms must be finalized in order to pass verify check
		llvm.DIBuilderFinalizeSubprogram(c.dibuilder, fn.subprogram)
	}

	return nil
}

func (c *Compiler) createInstruction(ctx context.Context, instr ssa.Instruction) (err error) {
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
			return err
		}

		b0 := c.blocks[instr.Block().Succs[0]]
		b1 := c.blocks[instr.Block().Succs[1]]

		llvm.BuildCondBr(c.builder, condValue.UnderlyingValue(ctx), b0, b1)
	case *ssa.Jump:
		if block, ok := c.blocks[instr.Block().Succs[0]]; ok {
			llvm.BuildBr(c.builder, block)
		} else {
			panic("block not created")
		}
	case *ssa.MapUpdate:
		panic("not implemented")
	case *ssa.Panic:
		arg, err := c.createExpression(ctx, instr.X)
		if err != nil {
			return err
		}
		_, err = c.createRuntimeCall(ctx, "panic", []llvm.LLVMValueRef{arg.UnderlyingValue(ctx)})
		if err != nil {
			return err
		}

		// Create an unreachable terminator following the panic
		llvm.BuildUnreachable(c.builder)
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
				return err
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
			llvm.BuildRet(c.builder, returnValue)
		} else {
			// Return nothing
			llvm.BuildRetVoid(c.builder)
		}
	case *ssa.RunDefers:
		fn, ok := c.functions[instr.Parent().Signature]
		if !ok {
			panic("function does not exist")
		}

		// The RunDefers instruction is always at the end of a function.
		// Therefore, we can track if a defer statement was created earlier in
		// order to determine if there is a defer stack that needs to be
		// processed. This primarily for when *ssa.NaiveForm is enabled.
		if fn.hasDefer {
			panic("not implemented")
		}
	case *ssa.Select:
		panic("not implemented")
	case *ssa.Send:
		panic("not implemented")
	case *ssa.Store:
		var err error
		var addr, value Value
		addr, err = c.createExpression(ctx, instr.Addr)
		if err != nil {
			return err
		}

		value, err = c.createExpression(ctx, instr.Val)
		if err != nil {
			return err
		}

		llvm.BuildStore(c.builder, value.UnderlyingValue(ctx), addr.UnderlyingValue(ctx))

		// NOTE: The value does not change if the address is on the heap since
		//       the value would be a pointer to a pointer. Instead, the
		//       value at the address of pointed-to pointer is change meaning
		//       no value change should be indicated below.
		if addr.dbg != nil && !addr.heap && !addr.global {
			// Attach debug information
			llvm.DIBuilderInsertDbgValueAtEnd(
				c.dibuilder,
				addr.UnderlyingValue(ctx),
				addr.dbg,
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
			_, err = c.createExpression(ctx, value)
			if err != nil {
				return err
			}
		} else {
			panic("encountered unknown instruction")
		}
	}

	return
}

func (c *Compiler) createFunctionCall(ctx context.Context, callee *ssa.Function, args []ssa.Value) (llvm.LLVMValueRef, error) {
	// Get the info about the callee
	fn, ok := c.functions[callee.Signature]
	if !ok {
		panic("call to function that does not exist")
	}

	// TODO: If this is a struct method call or an interface call, nil check the receiver.

	// Create the argument values
	values, err := c.createValues(ctx, args)
	if err != nil {
		return nil, err
	}

	// Create and return the value of the call
	return llvm.BuildCall2(c.builder, fn.llvmType, fn.value, values.Ref(ctx), ""), nil
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
			println(llvm.PrintTypeToString(argType))
			panic("argument type does not match")
		}
	}

	// Create and return the value of the call
	return llvm.BuildCall2(c.builder, fnType, fn, args, ""), nil
}

func (c *Compiler) positionAtEntryBlock(ctx context.Context) {
	entryBlock := c.currentEntryBlock(ctx)
	if blockFirst := llvm.GetFirstInstruction(entryBlock); blockFirst != nil {
		llvm.PositionBuilderBefore(c.builder, blockFirst)
	} else {
		llvm.PositionBuilderAtEnd(c.builder, entryBlock)
	}
}
