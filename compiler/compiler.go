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
	llvmContextKey        struct{}
	currentPackageKey     struct{}
	entryBlockKey         struct{}
	currentBlockKey       struct{}
	currentFnTypeKey      struct{}
	currentDbgLocationKey struct{}
)

type Function struct {
	value      llvm.LLVMValueRef
	llvmType   llvm.LLVMTypeRef
	def        *ssa.Function
	signature  *types.Signature
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

type instruction interface {
	Parent() *ssa.Function
	Pos() token.Pos
}

type Compiler struct {
	options          Options
	module           llvm.LLVMModuleRef
	builder          llvm.LLVMBuilderRef
	dibuilder        llvm.LLVMDIBuilderRef
	compileUnit      llvm.LLVMMetadataRef
	packageInitBlock llvm.LLVMValueRef
	functions        map[*types.Signature]*Function
	closures         map[*types.Signature]*Function
	signatures       map[*types.Signature]llvm.LLVMTypeRef
	types            map[types.Type]*Type
	descriptors      map[types.Type]llvm.LLVMValueRef
	values           map[ssa.Value]Value
	phis             map[*ssa.Phi]Phi
	blocks           map[*ssa.BasicBlock]llvm.LLVMBasicBlockRef
	uintptrType      Type
	ptrType          Type
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
		closures:    map[*types.Signature]*Function{},
		signatures:  map[*types.Signature]llvm.LLVMTypeRef{},
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

	// Create package initializer function
	pkgInitFnType := llvm.FunctionType(llvm.VoidTypeInContext(ctx), []llvm.LLVMTypeRef{}, false)
	pkgInitFn := llvm.AddFunction(module, "runtime.initPackages", pkgInitFnType)
	cc.packageInitBlock = llvm.AppendBasicBlockInContext(ctx, pkgInitFn, "pkg_init_entry")
	llvm.PositionBuilderAtEnd(builder, cc.packageInitBlock)
	llvm.BuildRetVoid(builder)

	cc.functions[types.NewSignatureType(nil, nil, nil, nil, nil, false)] = &Function{
		value:      pkgInitFn,
		llvmType:   pkgInitFnType,
		def:        nil,
		signature:  nil,
		diFile:     nil,
		subprogram: nil,
		compiled:   true,
		hasDefer:   false,
		name:       "runtime.initPackages",
	}

	// Create goroutine stack size constant
	constGoroutineStackSize := llvm.ConstInt(uintptrType.valueType, options.GoroutineStackSize, false)
	globalGoroutineStackSize := llvm.AddGlobal(module, llvm.TypeOf(constGoroutineStackSize), "runtime._goroutineStackSize")
	llvm.SetInitializer(globalGoroutineStackSize, constGoroutineStackSize)
	llvm.SetLinkage(globalGoroutineStackSize, llvm.LLVMLinkage(llvm.ExternalLinkage))
	llvm.SetGlobalConstant(globalGoroutineStackSize, true)

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

func (c *Compiler) currentDbgLocation(ctx context.Context) llvm.LLVMMetadataRef {
	if v := ctx.Value(currentDbgLocationKey{}); v != nil {
		return v.(llvm.LLVMMetadataRef)
	}
	return nil
}

func (c *Compiler) CompilePackage(ctx context.Context, llvmCtx llvm.LLVMContextRef, pkg *ssa.Package) error {
	var uncompiledFunctions []*Function

	// Set the LLVM context to the compiler context
	ctx = context.WithValue(ctx, llvmContextKey{}, llvmCtx)

	// Set the current package in the context
	ctx = context.WithValue(ctx, currentPackageKey{}, pkg)

	// Create any named types first
	for _, member := range pkg.Members {
		if ssaType, ok := member.(*ssa.Type); ok {
			c.createType(ctx, ssaType.Type())

			// Create any methods for this named type
			if namedType, ok := ssaType.Type().(*types.Named); ok {
				for i := 0; i < namedType.NumMethods(); i++ {
					method := namedType.Method(i)
					fn, err := c.createFunction(ctx, pkg.Prog.FuncValue(method))
					if err != nil {
						return err
					}
					c.functions[fn.def.Signature] = fn
					uncompiledFunctions = append(uncompiledFunctions, fn)
				}
			}
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

	// Create the type for each function first so that they are available when it's time to create the blocks.
	for _, member := range pkg.Members {
		if ssaFn, ok := member.(*ssa.Function); ok {
			fn, err := c.createFunction(ctx, ssaFn)
			if err != nil {
				return err
			}
			c.functions[ssaFn.Signature] = fn

			if len(ssaFn.Blocks) > 0 {
				uncompiledFunctions = append(uncompiledFunctions, fn)

				// Create anonymous functions
				for _, anonFn := range ssaFn.AnonFuncs {
					fn, err := c.createFunction(ctx, anonFn)
					if err != nil {
						return err
					}
					c.functions[anonFn.Signature] = fn
					if len(anonFn.FreeVars) > 0 {
						// This function is a closure
						c.closures[anonFn.Signature] = fn
					}
					uncompiledFunctions = append(uncompiledFunctions, fn)
				}

				if ssaFn.Name() == "init" {
					// Create a call to this package initializer
					llvm.PositionBuilderBefore(c.builder, llvm.GetLastInstruction(c.packageInitBlock))
					llvm.BuildCall2(c.builder, fn.llvmType, fn.value, []llvm.LLVMValueRef{}, "")
				}
			}
		}
	}

	// Finally create the blocks for each function
	for _, fn := range uncompiledFunctions {
		// Only attempt to compile functions that have blocks. This will allow
		// the compiler to support forward declarations of functions.
		if fn.def != nil && len(fn.def.Blocks) > 0 {
			if err := c.createFunctionBlocks(ctx, fn); err != nil {
				return err
			}
		}
	}

	return nil
}

func (c *Compiler) createFunction(ctx context.Context, fn *ssa.Function) (*Function, error) {
	if fn, ok := c.functions[fn.Signature]; ok {
		return fn, nil
	}

	isMethod := false
	receiver := fn.Signature.Recv()

	// Determine the function name. //go:linkname can override this
	name := ""
	if receiver != nil {
		typename := ""
		switch t := receiver.Type().(type) {
		case *types.Pointer:
			typename = t.Elem().(*types.Named).Obj().Name()
		case *types.Named:
			typename = t.Obj().Name()
		default:
			panic("unimplemented named type")
		}
		name = types.Id(c.currentPackage(ctx).Pkg, typename+"."+fn.Name())
		isMethod = true
	} else {
		name = types.Id(c.currentPackage(ctx).Pkg, fn.Name())
	}

	if len(name) == 0 {
		panic("function with no name")
	}

	// Process any pragmas
	info := c.options.GetSymbolInfo(name)
	if len(info.LinkName) > 0 {
		// Override the generated name
		name = info.LinkName
	}

	// Overriding type methods is not permitted
	if !isMethod {
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
	}

	// Cannot be both exported and external
	if info.Exported && info.ExternalLinkage {
		panic("function cannot be both external and exported")
	}

	// Create a new signature prepending the receiver and appending the free vars to the end of the of parameters
	var paramVars []*types.Var
	if fn.Signature.Recv() != nil {
		paramVars = append(paramVars, fn.Signature.Recv())
	}

	for i := 0; i < fn.Signature.Params().Len(); i++ {
		paramVars = append(paramVars, fn.Signature.Params().At(i))
	}

	for _, fv := range fn.FreeVars {
		paramVars = append(paramVars, types.NewVar(token.NoPos, fn.Pkg.Pkg, fv.Name(), fv.Type()))
	}

	var recvTypeParams []*types.TypeParam
	var typeParams []*types.TypeParam

	for i := 0; i < fn.Signature.RecvTypeParams().Len(); i++ {
		recvTypeParams = append(recvTypeParams, fn.Signature.RecvTypeParams().At(i))
	}

	for i := 0; i < fn.Signature.TypeParams().Len(); i++ {
		typeParams = append(typeParams, fn.Signature.TypeParams().At(i))
	}

	signature := types.NewSignatureType(
		fn.Signature.Recv(),
		recvTypeParams,
		typeParams,
		types.NewTuple(paramVars...),
		fn.Signature.Results(),
		fn.Signature.Variadic())

	// Create the function type
	// NOTE: A pointer type will be returned.
	c.createType(ctx, signature)

	// Get the function signature type from the signatures map
	fnType := c.signatures[signature]

	// Add the function to the current module
	fnValue := llvm.AddFunction(c.module, name, fnType)

	if !info.Exported {
		// Set export status
		llvm.SetVisibility(fnValue, llvm.LLVMVisibility(llvm.HiddenVisibility))
	}

	if !info.ExternalLinkage {
		llvm.SetLinkage(fnValue, llvm.LLVMLinkage(llvm.ExternalLinkage))
	}

	// Apply attributes
	if info.IsInterrupt {
		llvm.AddAttributeAtIndex(fnValue, uint(llvm.AttributeFunctionIndex), c.getAttribute(ctx, "noinline"))
		llvm.AddAttributeAtIndex(fnValue, uint(llvm.AttributeFunctionIndex), c.getAttribute(ctx, "noimplicitfloat"))
	}

	result := Function{
		value:     fnValue,
		llvmType:  fnType,
		def:       fn,
		signature: signature,
		name:      name,
	}

	return &result, nil
}

func (c *Compiler) createFunctionBlocks(ctx context.Context, fn *Function) error {
	// Panic now if this function was already compiled
	if fn.compiled {
		panic(fmt.Sprintf("multiple definitions of function \"%s\" exist", fn.def.Object().Id()))
	}

	c.println(Debug, "Compiling", fn.name)
	defer c.println(Debug, "Done compiling", fn.name)

	// Set the current function type in the context
	ctx = context.WithValue(ctx, currentFnTypeKey{}, fn)

	// Get the file information for this function
	file := fn.def.Prog.Fset.File(fn.def.Pos())

	// Some functions, like package initializers, do not have position information
	if file != nil {
		// Extract the file info
		filename := c.options.MapPath(file.Name())
		line := uint(file.Line(fn.def.Pos()))
		fnType := c.createDebugType(ctx, fn.signature)

		fn.diFile = llvm.DIBuilderCreateFile(
			c.dibuilder,
			filepath.Base(filename),
			filepath.Dir(filename))

		fn.subprogram = llvm.DIBuilderCreateFunction(
			c.dibuilder,
			fn.diFile,
			fn.name,
			fn.name,
			fn.diFile,
			line,
			fnType,
			true,
			true, 0, llvm.LLVMDIFlags(llvm.DIFlagPrototyped), false)

		// Apply this metadata to the function
		llvm.SetSubprogram(fn.value, fn.subprogram)
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

	// Get the current debug location to restore to when this instruction is done
	currentDbgLoc := c.currentDbgLocation(ctx)
	defer llvm.SetCurrentDebugLocation2(c.builder, currentDbgLoc)

	// Change the current debug location to that of the instruction being processed
	if instr.Parent() != nil {
		if file := instr.Parent().Prog.Fset.File(instr.Pos()); file != nil {
			if scope := c.instructionScope(instr); scope != nil {
				locationInfo := file.Position(instr.Pos())
				dbgLoc := llvm.DIBuilderCreateDebugLocation(
					c.currentContext(ctx),
					uint(locationInfo.Line),
					uint(locationInfo.Column),
					c.instructionScope(instr),
					nil,
				)
				ctx = context.WithValue(ctx, currentDbgLocationKey{}, dbgLoc)
				llvm.SetCurrentDebugLocation2(c.builder, dbgLoc)
			}
		}
	}

	// Create the specific instruction
	switch instr := instr.(type) {
	case *ssa.Defer:
		panic("not implemented")
	case *ssa.Go:
		callArgs := instr.Call.Args
		if closureExpr, ok := instr.Call.Value.(*ssa.MakeClosure); ok {
			callArgs = append(callArgs, closureExpr.Bindings...)
		}
		closure, args := c.createClosure(ctx, instr.Call.Signature(), callArgs)
		goroutineStructType := llvm.StructType([]llvm.LLVMTypeRef{c.ptrType.valueType, c.ptrType.valueType}, false)
		goroutineValue := llvm.BuildAlloca(c.builder, goroutineStructType, "goroutine")
		addr := llvm.BuildStructGEP2(c.builder, goroutineStructType, goroutineValue, 0, "goroutine_fn_ptr")
		llvm.BuildStore(c.builder, closure, addr)
		addr = llvm.BuildStructGEP2(c.builder, goroutineStructType, goroutineValue, 1, "goroutine_params_ptr")
		llvm.BuildStore(c.builder, args, addr)

		goroutineValue = llvm.BuildBitCast(c.builder, goroutineValue, c.ptrType.valueType, "")
		if _, err = c.createRuntimeCall(ctx, "addTask", []llvm.LLVMValueRef{goroutineValue}); err != nil {
			return err
		}
	case *ssa.If:
		condValue, err := c.createExpression(ctx, instr.Cond)
		if err != nil {
			return err
		}

		b0 := c.blocks[instr.Block().Succs[0]]
		b1 := c.blocks[instr.Block().Succs[1]]

		_ = llvm.BuildCondBr(c.builder, condValue.UnderlyingValue(ctx), b0, b1)
	case *ssa.Jump:
		if block, ok := c.blocks[instr.Block().Succs[0]]; ok {
			_ = llvm.BuildBr(c.builder, block)
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
			_ = llvm.BuildRet(c.builder, returnValue)
		} else {
			// Return nothing
			_ = llvm.BuildRetVoid(c.builder)
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

		_ = llvm.BuildStore(c.builder, value.UnderlyingValue(ctx), addr.UnderlyingValue(ctx))

		// NOTE: The value does not change if the address is on the heap since
		//       the value would be a pointer to a pointer. Instead, the
		//       value at the address of pointed-to pointer is change meaning
		//       no value change should be indicated below.
		if addr.dbg != nil && !addr.heap && !addr.global && instr.Pos().IsValid() {
			// Attach debug information
			llvm.DIBuilderInsertDbgValueAtEnd(
				c.dibuilder,
				addr.UnderlyingValue(ctx),
				addr.dbg,
				llvm.DIBuilderCreateExpression(c.dibuilder, nil),
				addr.DebugPos(ctx),
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

func (c *Compiler) instructionScope(instr instruction) llvm.LLVMMetadataRef {
	scope := instr.Parent().Pkg.Pkg.Scope().Innermost(instr.Pos())
	if scope == nil {
		panic("instruction has nil scope")
	}

	file := instr.Parent().Prog.Fset.File(instr.Pos())
	if file != nil {
		filename := c.options.MapPath(file.Name())
		diFile := llvm.DIBuilderCreateFile(
			c.dibuilder,
			filepath.Base(filename),
			filepath.Dir(filename))

		pos := file.Position(scope.Pos())
		fn := c.functions[instr.Parent().Signature]

		if fn.subprogram != nil {
			// Create a lexical block for the scope
			block := llvm.DIBuilderCreateLexicalBlock(
				c.dibuilder,
				fn.subprogram,
				diFile,
				uint(pos.Line),
				uint(pos.Column))

			return block
		}
	}
	return nil
}

func (c *Compiler) getAttribute(ctx context.Context, attr string) llvm.LLVMAttributeRef {
	return llvm.CreateEnumAttribute(c.currentContext(ctx), llvm.GetEnumAttributeKindForName(attr), 0)

}
