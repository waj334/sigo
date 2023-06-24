package compiler

import (
	"context"
	"fmt"
	"go/token"
	"go/types"
	"path/filepath"
	"strconv"

	"golang.org/x/tools/go/ssa"

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
	deferTop   llvm.LLVMValueRef
	llvmType   llvm.LLVMTypeRef
	stateType  llvm.LLVMTypeRef
	def        *ssa.Function
	signature  *types.Signature
	diFile     llvm.LLVMMetadataRef
	subprogram llvm.LLVMMetadataRef
	compiled   bool
	name       string
}

type Type struct {
	valueType  llvm.LLVMTypeRef
	debugType  llvm.LLVMMetadataRef
	spec       types.Type
	descriptor llvm.LLVMValueRef
}

func (t *Type) Nil() llvm.LLVMValueRef {
	return llvm.ConstNull(t.valueType)
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
	options         *Options
	module          llvm.LLVMModuleRef
	builder         llvm.LLVMBuilderRef
	dibuilder       llvm.LLVMDIBuilderRef
	compileUnit     llvm.LLVMMetadataRef
	functions       map[*ssa.Function]*Function
	closures        map[*types.Signature]*Function
	closureArgTypes map[*types.Func]llvm.LLVMTypeRef
	signatures      map[*types.Signature]llvm.LLVMTypeRef
	subprograms     map[*types.Signature]llvm.LLVMMetadataRef
	types           map[types.Type]*Type
	descriptors     map[types.Type]llvm.LLVMValueRef
	values          map[ssa.Value]Value
	phis            map[*ssa.Phi]Phi
	blocks          map[*ssa.BasicBlock]llvm.LLVMBasicBlockRef
	uintptrType     Type
	ptrType         Type
}

func NewCompiler(name string, options *Options) (*Compiler, llvm.LLVMContextRef) {
	// Create the LLVM context
	ctx := llvm.ContextCreate()

	// Create the module
	module := llvm.ModuleCreateWithNameInContext(name, ctx)
	dwarfVersion := llvm.ConstInt(llvm.Int32TypeInContext(ctx), 4, false)
	metadataVersion := llvm.ConstInt(llvm.Int32TypeInContext(ctx), uint64(llvm.DebugMetadataVersion()), false)
	llvm.AddModuleFlag(module, llvm.ModuleFlagBehaviorError, "Dwarf Version", llvm.ValueAsMetadata(dwarfVersion))
	llvm.AddModuleFlag(module, llvm.ModuleFlagBehaviorError, "Debug Info Version", llvm.ValueAsMetadata(metadataVersion))

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
		true,
		"",
		0,
		"",
		llvm.LLVMDWARFEmissionKind(llvm.DWARFEmissionFull),
		0,
		true,
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
		valueType: llvm.PointerTypeInContext(ctx, 0),
		debugType: llvm.DIBuilderCreateBasicType(dibuilder, "void*", size, DW_ATE_address, 0),
	}

	// Create and return the compiler
	cc := Compiler{
		options:         options,
		module:          module,
		builder:         builder,
		dibuilder:       dibuilder,
		compileUnit:     cu,
		functions:       map[*ssa.Function]*Function{},
		closures:        map[*types.Signature]*Function{},
		closureArgTypes: map[*types.Func]llvm.LLVMTypeRef{},
		signatures:      map[*types.Signature]llvm.LLVMTypeRef{},
		subprograms:     map[*types.Signature]llvm.LLVMMetadataRef{},
		types:           map[types.Type]*Type{},
		descriptors:     map[types.Type]llvm.LLVMValueRef{},
		values:          map[ssa.Value]Value{},
		blocks:          map[*ssa.BasicBlock]llvm.LLVMBasicBlockRef{},
		phis:            map[*ssa.Phi]Phi{},
		uintptrType:     uintptrType,
		ptrType:         ptrType,
	}

	return &cc, ctx
}

func (c *Compiler) Options() *Options {
	return c.options
}

func (c *Compiler) Module() llvm.LLVMModuleRef {
	return c.module
}

func (c *Compiler) Finalize() {
	// Resolve all debug metadata nodes
	llvm.DIBuilderFinalize(c.dibuilder)
}

func (c *Compiler) Dispose() {
	llvm.DisposeModule(c.module)
	llvm.DisposeDIBuilder(c.dibuilder)
	llvm.DisposeBuilder(c.builder)
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
					ssaFn := pkg.Prog.FuncValue(method)
					fn := c.createFunction(ctx, ssaFn)
					uncompiledFunctions = append(uncompiledFunctions, fn)
				}
			}
		}
	}

	// Create constants and globals next
	for _, member := range pkg.Members {
		switch member := member.(type) {
		case *ssa.NamedConst:
			c.createExpression(ctx, member.Value)
		case *ssa.Global:
			value := c.createExpression(ctx, member)
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

			llvm.GlobalSetMetadata(value.UnderlyingValue(ctx), llvm.GetMDKindID("dbg"), value.dbg)

			// Cache the global
			c.values[member] = value
		}
	}

	// Create the type for each function first so that they are available when it's time to create the blocks.
	for _, member := range pkg.Members {
		if ssaFn, ok := member.(*ssa.Function); ok {
			if c.isIntrinsic(ssaFn) {
				// Skip intrinsic functions
				continue
			}

			fn := c.createFunction(ctx, ssaFn)

			if len(ssaFn.Blocks) > 0 {
				uncompiledFunctions = append(uncompiledFunctions, fn)

				// Create anonymous functions
				for _, anonFn := range ssaFn.AnonFuncs {
					fn := c.createFunction(ctx, anonFn)
					if len(anonFn.FreeVars) > 0 {
						// This function is a closure
						c.closures[anonFn.Signature] = fn
					}
					uncompiledFunctions = append(uncompiledFunctions, fn)
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

func (c *Compiler) createFunction(ctx context.Context, fn *ssa.Function) *Function {
	if f, ok := c.functions[fn]; ok {
		return f
	}

	isMethod := false
	receiver := fn.Signature.Recv()

	// Determine the function name. //go:linkname can override this
	symbolName := ""
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
		symbolName = c.symbolName(fn.Pkg.Pkg, typename+"."+fn.Name())
		isMethod = true
	} else {
		symbolName = c.symbolName(fn.Pkg.Pkg, fn.Name())
	}

	name := symbolName

	// Process any pragmas
	info := c.options.GetSymbolInfo(symbolName)
	if len(info.LinkName) > 0 && !isMethod {
		// Override the generated name
		name = info.LinkName
	}

	if len(name) == 0 {
		panic("function with no name")
	}

	var fnValue llvm.LLVMValueRef
	var stateType llvm.LLVMTypeRef

	// Create the function type
	fnType := c.createFunctionType(ctx, fn.Signature, len(fn.FreeVars) > 0)
	c.signatures[fn.Signature] = fnType

	if len(fn.FreeVars) > 0 {
		// Collect the types of the free vars
		var freeVarTypes []llvm.LLVMTypeRef
		for _, fv := range fn.FreeVars {
			freeVarTypes = append(freeVarTypes, c.createType(ctx, fv.Type()).valueType)
		}

		// Create the struct to hold the free vars
		stateType = llvm.StructTypeInContext(c.currentContext(ctx), freeVarTypes, false)
	}

	// Add the function to the current module
	fnValue = llvm.AddFunction(c.module, name, fnType)

	// Cannot be both exported and external
	if info.Exported && info.ExternalLinkage {
		println(symbolName)
		panic("function cannot be both external and exported")
	}

	isExported := false
	if obj := fn.Object(); obj != nil && obj.Exported() {
		isExported = true
	} else if info.Exported {
		isExported = true
	}
	//isWeak := false
	if !isExported {
		llvm.SetVisibility(fnValue, llvm.HiddenVisibility)
	} else if info.ExternalLinkage {
		llvm.SetLinkage(fnValue, llvm.ExternalLinkage)
	}

	// Apply attributes
	if info.IsInterrupt {
		llvm.AddAttributeAtIndex(fnValue, uint(llvm.AttributeFunctionIndex), c.getAttribute(ctx, "noinline"))
		llvm.AddAttributeAtIndex(fnValue, uint(llvm.AttributeFunctionIndex), c.getAttribute(ctx, "noimplicitfloat"))
	}

	var diFile llvm.LLVMMetadataRef
	var subprogram llvm.LLVMMetadataRef

	if !info.ExternalLinkage && fn.Pkg == c.currentPackage(ctx) {
		// Get the file information for this function
		file := fn.Prog.Fset.File(fn.Pos())

		// Some functions, like package initializers, do not have position information
		line := uint(0)
		if file != nil {
			// Extract the file info
			filename := c.options.MapPath(file.Name())
			line = uint(file.Line(fn.Pos()))
			diFile = llvm.DIBuilderCreateFile(
				c.dibuilder,
				filepath.Base(filename),
				filepath.Dir(filename))
		} else {
			diFile = llvm.DIBuilderCreateFile(
				c.dibuilder,
				"<unknown>",
				"<unknown>")
		}

		subprogram = llvm.DIBuilderCreateFunction(
			c.dibuilder,
			diFile,
			name,
			name,
			diFile,
			line,
			c.createDebugType(ctx, fn.Signature),
			fn.Pkg == c.currentPackage(ctx),
			fn.Pkg == c.currentPackage(ctx),
			line,
			llvm.LLVMDIFlags(llvm.DIFlagPrototyped), false)

		// Subprograms must be finalized in order to pass verify check
		llvm.DIBuilderFinalizeSubprogram(c.dibuilder, subprogram)

		// Apply this metadata to the function
		llvm.SetSubprogram(fnValue, subprogram)
		c.subprograms[fn.Signature] = subprogram
	}

	result := &Function{
		value:      fnValue,
		llvmType:   fnType,
		stateType:  stateType,
		def:        fn,
		signature:  fn.Signature,
		name:       name,
		subprogram: subprogram,
		diFile:     diFile,
	}
	c.functions[fn] = result
	return result
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

	return nil
}

func (c *Compiler) createInstruction(ctx context.Context, instr ssa.Instruction) (err error) {
	c.printf(Debug, "Processing instruction %T: %s\n", instr, instr.String())

	// Get the current debug location to restore to when this instruction is done
	currentDbgLoc := c.currentDbgLocation(ctx)
	defer llvm.SetCurrentDebugLocation2(c.builder, currentDbgLoc)

	// Change the current debug location to that of the instruction being processed
	scope, file := c.instructionScope(instr)
	var location token.Position
	if file != nil {
		location = file.Position(instr.Pos())
	}
	dbgLoc := llvm.DIBuilderCreateDebugLocation(
		c.currentContext(ctx),
		uint(location.Line),
		uint(location.Column),
		scope,
		nil)

	ctx = context.WithValue(ctx, currentDbgLocationKey{}, dbgLoc)
	llvm.SetCurrentDebugLocation2(c.builder, dbgLoc)

	// Create the specific instruction
	switch instr := instr.(type) {
	case *ssa.Defer:
		fn := c.currentFunction(ctx)
		if fn.deferTop == nil {
			// Allocate the pointer to current top of the defer stack
			fn.deferTop = c.createAlloca(ctx, c.ptrType.valueType, "defer_top")

			// Store the top onto the stack for the current call
			llvm.BuildStore(c.builder, fn.deferTop, c.ptrType.Nil())
		}

		// Create a closure for the defer call
		var closure llvm.LLVMValueRef
		var closureContextType llvm.LLVMTypeRef
		var args []llvm.LLVMValueRef

		switch value := instr.Call.Value.(type) {
		case *ssa.Function:
			// Create a new closure
			closureContextType = c.createClosureContextType(ctx, value)
			closure = c.createClosure(ctx, value, closureContextType, true)
			args = c.createValues(ctx, instr.Call.Args).Ref(ctx)
		case *ssa.MakeClosure:
			state := c.createExpression(ctx, value).UnderlyingValue(ctx)
			closureContextType = c.createClosureContextType(ctx, value.Fn.(*ssa.Function))
			closure = c.createClosure(ctx, value.Fn.(*ssa.Function), closureContextType, true)
			args = append(c.createValues(ctx, instr.Call.Args).Ref(ctx), state)
		default:
			panic("unhandled")
		}

		c.createExpression(ctx, instr.Call.Value)
		callArgs := instr.Call.Args

		closureCtx := c.createClosureContext(ctx, closureContextType, args)
		if closureExpr, ok := instr.Call.Value.(*ssa.MakeClosure); ok {
			callArgs = append(callArgs, closureExpr.Bindings...)
		}

		// Push the defer frame to the defer stack
		c.createRuntimeCall(ctx, "deferPush", []llvm.LLVMValueRef{
			closure,
			closureCtx,
			fn.deferTop,
		})
	case *ssa.Go:
		var closure llvm.LLVMValueRef
		var closureContextType llvm.LLVMTypeRef
		var args []llvm.LLVMValueRef

		switch value := instr.Call.Value.(type) {
		case *ssa.Function:
			// Create a new closure
			closureContextType = c.createClosureContextType(ctx, value)
			closure = c.createClosure(ctx, value, closureContextType, true)
			args = c.createValues(ctx, instr.Call.Args).Ref(ctx)
		case *ssa.MakeClosure:
			state := c.createExpression(ctx, value).UnderlyingValue(ctx)
			closureContextType = c.createClosureContextType(ctx, value.Fn.(*ssa.Function))
			closure = c.createClosure(ctx, value.Fn.(*ssa.Function), closureContextType, true)
			args = append(c.createValues(ctx, instr.Call.Args).Ref(ctx), state)
		default:
			panic("unhandled")
		}

		c.createExpression(ctx, instr.Call.Value)
		callArgs := instr.Call.Args

		closureCtx := c.createClosureContext(ctx, closureContextType, args)
		if closureExpr, ok := instr.Call.Value.(*ssa.MakeClosure); ok {
			callArgs = append(callArgs, closureExpr.Bindings...)
		}

		goroutineStructType := llvm.StructType([]llvm.LLVMTypeRef{c.ptrType.valueType, c.ptrType.valueType}, false)
		goroutineValue := c.createAlloca(ctx, goroutineStructType, "goroutine")
		addr := llvm.BuildStructGEP2(c.builder, goroutineStructType, goroutineValue, 0, "goroutine_fn_ptr")
		llvm.BuildStore(c.builder, closure, addr)
		addr = llvm.BuildStructGEP2(c.builder, goroutineStructType, goroutineValue, 1, "goroutine_params_ptr")
		llvm.BuildStore(c.builder, closureCtx, addr)

		goroutineValue = llvm.BuildBitCast(c.builder, goroutineValue, c.ptrType.valueType, "")
		c.createRuntimeCall(ctx, "addTask", []llvm.LLVMValueRef{goroutineValue})
	case *ssa.If:
		condValue := c.createExpression(ctx, instr.Cond).UnderlyingValue(ctx)

		b0 := c.blocks[instr.Block().Succs[0]]
		b1 := c.blocks[instr.Block().Succs[1]]

		_ = llvm.BuildCondBr(c.builder, condValue, b0, b1)
	case *ssa.Jump:
		if block, ok := c.blocks[instr.Block().Succs[0]]; ok {
			_ = llvm.BuildBr(c.builder, block)
		} else {
			panic("block not created")
		}
	case *ssa.MapUpdate:
		mapValue := c.createExpression(ctx, instr.Map).UnderlyingValue(ctx)
		key := c.createExpression(ctx, instr.Key).UnderlyingValue(ctx)
		value := c.createExpression(ctx, instr.Value).UnderlyingValue(ctx)

		// Create runtime call to update the map
		c.createRuntimeCall(ctx, "mapUpdate", []llvm.LLVMValueRef{
			mapValue,
			c.addressOf(ctx, key),
			c.addressOf(ctx, value),
		})
	case *ssa.Panic:
		fn := c.currentFunction(ctx)
		arg := c.createExpression(ctx, instr.X).UnderlyingValue(ctx)
		c.createRuntimeCall(ctx, "_panic", []llvm.LLVMValueRef{arg})

		// First, run defers if there are any
		if fn.deferTop != nil {
			// Get the top at the time for the current defer
			newTop := c.createRuntimeCall(ctx, "deferInitialTop", nil)

			// Load the pointer to store top to
			ptr := llvm.BuildLoad2(c.builder, c.ptrType.valueType, fn.deferTop, "")

			// Update top for this function
			llvm.BuildStore(c.builder, newTop, ptr)

			recoverBlock := c.blocks[fn.def.Blocks[len(fn.def.Blocks)-1]]

			// Jump to recover block
			llvm.BuildBr(c.builder, recoverBlock)
		} else {
			// Create an unreachable terminator following the panic
			llvm.BuildUnreachable(c.builder)
		}
	case *ssa.Return:
		fn := c.currentFunction(ctx)

		// Get the return type of this function
		returnType := llvm.GetReturnType(c.currentFunction(ctx).llvmType)

		// First, run defers if there are any
		if fn.deferTop != nil {
			c.createRuntimeCall(ctx, "deferRun", []llvm.LLVMValueRef{fn.deferTop})
		}

		// Void types do not need to return anything
		if llvm.GetTypeKind(returnType) != llvm.VoidTypeKind {
			// Get the return values
			returnValues := c.createValues(ctx, instr.Results).Ref(ctx)

			// Return a tuple if there should be more than one return value.
			// Otherwise, just return the single value.
			if len(instr.Results) > 1 {
				llvm.BuildAggregateRet(c.builder, returnValues)
			} else {
				// Return the single value
				llvm.BuildRet(c.builder, returnValues[0])
			}
		} else {
			// Return nothing
			llvm.BuildRetVoid(c.builder)
		}
	case *ssa.RunDefers:
		// Will not implement
	case *ssa.Select:
		panic("not implemented")
	case *ssa.Send:
		panic("not implemented")
	case *ssa.Store:
		addr := c.createExpression(ctx, instr.Addr)
		value := c.createExpression(ctx, instr.Val)
		llvm.BuildStore(c.builder, value.UnderlyingValue(ctx), addr.UnderlyingValue(ctx))

		// NOTE: The value does not change if the address is on the heap since
		//       the value would be a pointer to a pointer. Instead, the
		//       value at the address of pointed-to pointer is change meaning
		//       no value change should be indicated below.
		if addr.dbg != nil && instr.Pos().IsValid() && !addr.global { //&& !addr.heap && !addr.global && instr.Pos().IsValid() {
			// Attach debug information
			llvm.DIBuilderInsertDbgValueAtEnd(
				c.dibuilder,
				addr.ref,
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
			c.createExpression(ctx, value)
		} else {
			panic("encountered unknown instruction")
		}
	}

	return
}

func (c *Compiler) createFunctionCall(ctx context.Context, callee *ssa.Function, args []ssa.Value) llvm.LLVMValueRef {
	// Get the info about the callee
	fn, ok := c.functions[callee]
	if !ok {
		panic("call to function that does not exist")
	}

	// TODO: If this is a struct method call or an interface call, nil check the receiver.

	// Create the argument values
	values := c.createValues(ctx, args)

	argValues := values.Ref(ctx)

	// Process each argument value to make sure they are the correct types as
	// defined by the function.
	for i, t := range llvm.GetParamTypes(fn.llvmType) {
		argType := llvm.TypeOf(argValues[i])
		if !llvm.TypeIsEqual(argType, t) {
			println(llvm.PrintTypeToString(fn.llvmType))
			println(i, " expected: ", llvm.PrintTypeToString(t))
			println(i, " actual: ", llvm.PrintTypeToString(argType))
			panic("argument type does not match")
		}
	}

	// Create and return the value of the call
	return llvm.BuildCall2(c.builder, fn.llvmType, fn.value, argValues, "")
}

func (c *Compiler) createRuntimeCall(ctx context.Context, name string, args []llvm.LLVMValueRef) llvm.LLVMValueRef {
	c.println(Debug, "Creating runtime call:", name)

	// Get the function from the runtime package
	fn := c.currentPackage(ctx).Prog.ImportedPackage("runtime").Func(name)
	if fn == nil {
		panic("runtime does not implement " + name)
	}

	value := c.createExpression(ctx, fn).UnderlyingValue(ctx)
	fnType := c.signatures[fn.Signature]
	if fnType == nil {
		panic("function not cached?")
	}

	// Process each argument value to make sure they are the correct types as
	// defined by the function.
	for i, t := range llvm.GetParamTypes(fnType) {
		argType := llvm.TypeOf(args[i])
		if !llvm.TypeIsEqual(argType, t) {
			println("runtime.", name)
			println("expected: ", llvm.PrintTypeToString(t))
			println("actual: ", llvm.PrintTypeToString(argType))
			panic("argument type " + strconv.Itoa(i) + " does not match")
		}
	}

	// Create and return the value of the call
	return llvm.BuildCall2(c.builder, fnType, value, args, "")
}

func (c *Compiler) positionAtEntryBlock(ctx context.Context) {
	entryBlock := c.currentEntryBlock(ctx)
	if blockFirst := llvm.GetFirstInstruction(entryBlock); blockFirst != nil {
		llvm.PositionBuilderBefore(c.builder, blockFirst)
	} else {
		llvm.PositionBuilderAtEnd(c.builder, entryBlock)
	}
}

func (c *Compiler) instructionScope(instr instruction) (_ llvm.LLVMMetadataRef, file *token.File) {
	if instr.Parent() != nil {
		if instr.Parent().Object() != nil {
			fn := c.functions[instr.Parent()]
			if file = instr.Parent().Prog.Fset.File(instr.Pos()); file != nil {
				scope := instr.Parent().Pkg.Pkg.Scope().Innermost(instr.Pos())
				location := file.Position(scope.Pos())
				filename := c.options.MapPath(file.Name())
				return llvm.DIBuilderCreateLexicalBlock(
					c.dibuilder,
					fn.subprogram,
					llvm.DIBuilderCreateFile(
						c.dibuilder,
						filepath.Base(filename),
						filepath.Dir(filename)),
					uint(location.Line),
					uint(location.Column)), file
			} else {
				return fn.subprogram, nil
			}
		} else if subprogram, ok := c.subprograms[instr.Parent().Signature]; ok {
			return subprogram, nil
		}
	}
	return c.compileUnit, nil
}

func (c *Compiler) getAttribute(ctx context.Context, attr string) llvm.LLVMAttributeRef {
	return llvm.CreateEnumAttribute(c.currentContext(ctx), llvm.GetEnumAttributeKindForName(attr), 0)
}

func (c *Compiler) CreateInitLib(llctx llvm.LLVMContextRef, pkgs []*ssa.Package) {
	ctx := context.WithValue(context.Background(), llvmContextKey{}, llctx)

	// Create package initializer function
	pkgInitFnType := llvm.FunctionType(llvm.VoidTypeInContext(llctx), []llvm.LLVMTypeRef{}, false)
	packageInitFunc := llvm.AddFunction(c.module, "runtime.initPackages", pkgInitFnType)
	packageInitBlock := llvm.AppendBasicBlockInContext(llctx, packageInitFunc, "pkg_init_entry")
	llvm.PositionBuilderAtEnd(c.builder, packageInitBlock)

	// Create calls to each package init function
	for _, pkg := range pkgs {
		initFunc := pkg.Func("init")
		if initFunc != nil {
			linkname := c.symbolName(initFunc.Pkg.Pkg, initFunc.Name())
			fnType := c.createFunctionType(ctx, initFunc.Signature, false)
			fn := llvm.AddFunction(c.module, linkname, fnType)
			llvm.BuildCall2(c.builder, fnType, fn, nil, "")
		}
	}

	llvm.BuildRetVoid(c.builder)

	// Create goroutine stack size constant
	constGoroutineStackSize := llvm.ConstInt(c.uintptrType.valueType, c.options.GoroutineStackSize, false)
	globalGoroutineStackSize := llvm.AddGlobal(c.module, llvm.TypeOf(constGoroutineStackSize), "runtime._goroutineStackSize")
	llvm.SetInitializer(globalGoroutineStackSize, constGoroutineStackSize)
	llvm.SetLinkage(globalGoroutineStackSize, llvm.LLVMLinkage(llvm.ExternalLinkage))
	llvm.SetGlobalConstant(globalGoroutineStackSize, true)
}
