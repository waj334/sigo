package ssa

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"

	"golang.org/x/exp/maps"
	"golang.org/x/exp/slices"
	"golang.org/x/tools/go/packages"
	"pkg.si-go.dev/go-mlir/mlir"
	"pkg.si-go.dev/sigo/goir/binding/goir"
)

type Config struct {
	Fset               *token.FileSet
	Ctx                mlir.Context
	Module             mlir.Module
	Sizes              *types.StdSizes
	Program            *Program
	NumWorkers         int
	DisableUseAnalysis bool
}

type thunkType struct {
	t goir.FunctionType
	s *types.Signature
}

type Builder struct {
	ctx mlir.Context

	// Thread-safe structures (No async writes):
	config       Config
	program      *Program
	genericFuncs map[string]*funcData
	diFiles      map[*token.File]mlir.LLVMDIFileAttr
	compileUnits map[*token.File]mlir.LLVMDICompileUnitAttr
	symbols      mlir.SymbolTable
	declInfo     map[ast.Decl]*types.Info

	// Thread-unsafe structures:
	funcDeclDataMutex sync.RWMutex
	funcDeclData      map[string]*funcData
	ungeneratedFuncs  map[string]*ast.FuncDecl

	typeCache      map[types.Type]mlir.TypeLike
	typeCacheMutex sync.Mutex

	valueCache      map[types.Object]Value
	valueCacheMutex sync.RWMutex

	addToModuleMutex sync.Mutex
	addToModule      map[string]mlir.Operation

	forwardDeclarationsMutex sync.Mutex
	forwardDeclarations      map[string]mlir.Operation

	thunkMutex sync.Mutex
	thunks     map[string]struct{}
	thunkTypes map[string]thunkType

	trampolines     map[string]struct{}
	trampolineMutex sync.Mutex

	// Pending hooks keyed by *ast.FuncLit. Set by synthetic-closure callers
	// (e.g. range-over-func) so emitFuncLiteral can install body hooks even
	// when the FuncLit is consumed indirectly via emitCallArgs/emitExpr,
	// which doesn't propagate the variadic `setup` parameter.
	funcLitHooksMutex sync.Mutex
	funcLitHooks      map[*ast.FuncLit]func(*funcData)

	builtinWrapperMutex sync.Mutex
	builtinWrappers     map[string]string

	// Misc:
	generateQueue      chan *funcData
	work               atomic.Int32
	generateQueueMutex sync.Mutex

	i1    goir.BooleanType
	si    goir.IntegerType
	si8   goir.IntegerType
	si32  goir.IntegerType
	si64  goir.IntegerType
	ui    goir.IntegerType
	ui8   goir.IntegerType
	ui32  goir.IntegerType
	ui64  goir.IntegerType
	f32   mlir.FloatType
	f64   mlir.FloatType
	c64   mlir.ComplexType
	c128  mlir.ComplexType
	ptr   goir.UnsafePointerType
	uiptr goir.IntegerType
	str   goir.StringType

	_chan      mlir.TypeLike
	_map       mlir.TypeLike
	_slice     mlir.TypeLike
	_string    mlir.TypeLike
	_interface mlir.TypeLike
	_any       mlir.TypeLike
	_func      mlir.TypeLike
	_funcPtr   mlir.TypeLike

	_noLoc mlir.LocationLike

	initPackageCounter map[*packages.Package]*atomic.Uint32
}

type TypeParamMap map[int]types.Type

func NewBuilder(config Config) *Builder {
	builder := &Builder{
		config:          config,
		typeCache:       map[types.Type]mlir.TypeLike{},
		valueCache:      map[types.Object]Value{},
		ctx:             config.Ctx,
		program:         config.Program,
		symbols:         mlir.NewSymbolTable(config.Module.Operation()),
		generateQueue:   make(chan *funcData),
		thunks:          map[string]struct{}{},
		thunkTypes:      map[string]thunkType{},
		genericFuncs:    map[string]*funcData{},
		builtinWrappers: map[string]string{},
		trampolines:     map[string]struct{}{},

		ungeneratedFuncs:    map[string]*ast.FuncDecl{},
		funcDeclData:        map[string]*funcData{},
		addToModule:         map[string]mlir.Operation{},
		forwardDeclarations: map[string]mlir.Operation{},

		diFiles:            map[*token.File]mlir.LLVMDIFileAttr{},
		compileUnits:       map[*token.File]mlir.LLVMDICompileUnitAttr{},
		initPackageCounter: map[*packages.Package]*atomic.Uint32{},
		declInfo:           map[ast.Decl]*types.Info{},
	}

	// Create all basic types up front for ease of use later.
	builder.i1 = builder.GetType(context.Background(), types.Typ[types.Bool]).(goir.BooleanType)
	builder.si = builder.GetType(context.Background(), types.Typ[types.Int]).(goir.IntegerType)
	builder.si8 = builder.GetType(context.Background(), types.Typ[types.Int8]).(goir.IntegerType)
	builder.si32 = builder.GetType(context.Background(), types.Typ[types.Int32]).(goir.IntegerType)
	builder.si64 = builder.GetType(context.Background(), types.Typ[types.Int64]).(goir.IntegerType)
	builder.ui = builder.GetType(context.Background(), types.Typ[types.Uint]).(goir.IntegerType)
	builder.ui8 = builder.GetType(context.Background(), types.Typ[types.Uint8]).(goir.IntegerType)
	builder.ui32 = builder.GetType(context.Background(), types.Typ[types.Uint32]).(goir.IntegerType)
	builder.ui64 = builder.GetType(context.Background(), types.Typ[types.Uint64]).(goir.IntegerType)
	builder.f32 = builder.GetType(context.Background(), types.Typ[types.Float32]).(mlir.FloatType)
	builder.f64 = builder.GetType(context.Background(), types.Typ[types.Float64]).(mlir.FloatType)
	builder.c64 = builder.GetType(context.Background(), types.Typ[types.Complex64]).(mlir.ComplexType)
	builder.c128 = builder.GetType(context.Background(), types.Typ[types.Complex128]).(mlir.ComplexType)
	builder.ptr = builder.GetType(context.Background(), types.Typ[types.UnsafePointer]).(goir.UnsafePointerType)
	builder.uiptr = builder.GetType(context.Background(), types.Typ[types.Uintptr]).(goir.IntegerType)
	builder.str = builder.GetType(context.Background(), types.Typ[types.String]).(goir.StringType)

	builder._chan = builder.GetType(context.Background(), config.Program.LookupType("runtime", "_channel"))
	builder._interface = builder.GetType(context.Background(), config.Program.LookupType("runtime", "_interface"))
	builder._map = builder.GetType(context.Background(), config.Program.LookupType("runtime", "_map"))
	builder._slice = builder.GetType(context.Background(), config.Program.LookupType("runtime", "_slice"))
	builder._string = builder.GetType(context.Background(), config.Program.LookupType("runtime", "_string"))
	builder._func = builder.GetType(context.Background(), config.Program.LookupType("runtime", "_func"))
	builder._funcPtr = builder.GetType(context.Background(), types.NewPointer(config.Program.LookupType("runtime", "_func")))
	builder._any = builder.GetType(context.Background(), types.NewInterfaceType(nil, nil).Complete())

	// Bind the runtime type representations to the dialect's primitive type representation.
	// NOTE: The specific type does not matter since DLTI relies on a type's type ID which is the same each variation of
	//       a specific type in MLIR.
	goir.BindRuntimeTypeToType(config.Module, goir.NewChanType(builder.i1, goir.ChanDirectionRecvOnly), builder._chan)
	goir.BindRuntimeTypeToType(config.Module, builder._any, builder._interface)
	goir.BindRuntimeTypeToType(config.Module, goir.NewMapType(builder.i1, builder.i1), builder._map)
	goir.BindRuntimeTypeToType(config.Module, goir.NewSliceType(builder.i1), builder._slice)
	goir.BindRuntimeTypeToType(config.Module, builder.str, builder._string)

	builder._noLoc = builder.unscopedLocation(0)

	// Bind dialect types to runtime types.
	typeMap := map[string]string{
		"chan":                "runtime._channel",
		"interface":           "runtime._interface",
		"map":                 "runtime._map",
		"slice":               "runtime._slice",
		"string":              "runtime._string",
		"type":                "runtime._type",
		"func":                "runtime._func",
		"namedTypeData":       "runtime._namedTypeData",
		"funcData":            "runtime._funcData",
		"interfaceData":       "runtime._interfaceData",
		"interfaceMethodData": "runtime._interfaceMethodData",
		"signatureTypeData":   "runtime._signatureTypeData",
		"arrayTypeData":       "runtime._arrayTypeData",
		"structTypeData":      "runtime._structTypeData",
		"structFieldData":     "runtime._structFieldData",
		"channelTypeData":     "runtime._channelTypeData",
		"mapTypeData":         "runtime._mapTypeData",
	}

	for k, v := range typeMap {
		Tstr := strings.Split(v, ".")
		T := config.Program.LookupType(Tstr[0], Tstr[1])
		goir.BindRuntimeType(config.Module, k, builder.GetType(context.Background(), T))
	}

	return builder
}

func (b *Builder) GeneratePackages(ctx context.Context, pkgs []*packages.Package) {
	// All operations should go to the module body by default.
	ctx = newContextWithCurrentBlock(ctx)
	moduleRegion := b.config.Module.Operation().Region(0)
	ctx = newContextWithRegion(ctx, moduleRegion)
	moduleBlock := b.config.Module.Body()
	setCurrentBlock(ctx, moduleBlock)

	// Create a new job queue for when functions need other functions to be generated.
	queue := newJobQueue(1024)
	ctx = context.WithValue(ctx, jobQueueKey{}, queue)

	// Create debug information for each file.
	producerAttr := mlir.NewStringAttr(b.ctx, "SiGo")
	b.config.Fset.Iterate(func(file *token.File) bool {
		fname := file.Name()
		if evalPath, err := filepath.EvalSymlinks(fname); err == nil {
			fname = evalPath
		}

		nameAttr := mlir.NewStringAttr(b.ctx, filepath.Base(fname))
		fnameAttr := mlir.NewStringAttr(b.ctx, filepath.Dir(fname))
		diFileAttr := mlir.NewLLVMDIFileAttr(b.ctx, nameAttr, fnameAttr)
		b.diFiles[file] = diFileAttr

		// Create a matching compile unit for this file.
		idAttr := goir.NewDistinctAttr(fnameAttr)
		compileUnitAttr := mlir.NewLLVMDICompileUnitAttr(
			b.ctx,
			idAttr,
			mlir.LLVMDWARFSourceLanguageC,
			diFileAttr,
			producerAttr,
			false,
			mlir.LLVMDIEmissionKindFull,
			false,
			mlir.LLVMDINameTableKindDefault,
			b.strAttr(""),
			nil,
		)
		b.compileUnits[file] = compileUnitAttr
		return true
	})

	// Map declarations to their respective type checked info.
	for _, pkg := range pkgs {
		for _, file := range pkg.Syntax {
			for _, decl := range file.Decls {
				b.declInfo[decl] = pkg.TypesInfo
			}
		}
	}

	// Populate un-generated functions list first so global initializers can add work to the initial job queue.
	for _, pkg := range pkgs {
		ctx = newContextWithInfo(ctx, pkg.TypesInfo)
		for _, file := range pkg.Syntax {
			for _, decl := range file.Decls {
				if decl, ok := decl.(*ast.FuncDecl); ok {
					obj := b.objectOf(ctx, decl.Name).(*types.Func)
					symbol := qualifiedFuncName(obj)
					if symbol == b.config.Program.MainFunc {
						symbol = "main.main"
					}

					// Is this function an intrinsic?
					if isIntrinsic(symbol) {
						// Skip intrinsic functions.
						continue
					}

					// Mark this function as un-generated.
					b.ungeneratedFuncs[symbol] = decl
				}
			}
		}
	}

	// Emit constants and global variables serially.
	gvars := map[types.Object]*GlobalValue{}
	for _, pkg := range pkgs {
		ctx = newContextWithInfo(ctx, pkg.TypesInfo)
		for _, file := range pkg.Syntax {
			for _, decl := range file.Decls {
				switch decl := decl.(type) {
				case *ast.GenDecl:
					switch decl.Tok {
					case token.VAR:
						for _, spec := range decl.Specs {
							spec := spec.(*ast.ValueSpec)
							for _, ident := range spec.Names {
								if ident.Name == "_" {
									// Don't actually emit a global that does not have name. These are commonly used
									// to enforce a type check during parsing.
									continue
								}
								gvar := b.emitGlobalVar(ctx, ident)
								gvars[b.objectOf(ctx, ident)] = gvar
							}
						}
					default:
						b.emitDecl(ctx, decl)
					}
				}
			}
		}
	}

	// Handle global initializers.
	// NOTE: Need to range over the package list so each global is assigned a priority based on the dependency ordering
	//       of the packages.
	initializedGlobals := map[*GlobalValue]struct{}{}
	globalPriority := 0
	for _, pkg := range pkgs {
		for _, initializer := range pkg.TypesInfo.InitOrder {
			if len(initializer.Lhs) == 1 {
				lhs := initializer.Lhs[0]
				if lhs.Name() == "_" {
					// Don't actually emit a global that does not have name. These are commonly used
					// to enforce a type check during parsing.
					continue
				}

				gv := gvars[lhs]
				location := b.location(ctx, lhs.Pos())

				// Initialize this global.
				gv.Initialize(ctx, b, globalPriority, func(ctx context.Context, builder *Builder) mlir.Value {
					ctx = newContextWithInfo(ctx, pkg.TypesInfo)
					rhsType := b.typeOf(ctx, initializer.Rhs)

					result := b.emitExpr(ctx, initializer.Rhs)[0]

					if rhsType != nil {
						switch baseType(lhs.Type()).(type) {
						case *types.Interface:
							if !types.Identical(lhs.Type(), rhsType) {
								switch rhsType.(type) {
								case *types.Interface:
									result = b.emitChangeType(ctx, lhs.Type(), result, location)
								default:
									result = b.emitInterfaceValue(ctx, lhs.Type(), rhsType, result, location)
								}
							}
						default:
							switch b.typeOf(ctx, initializer.Rhs).(type) {
							case *types.Signature:
								// Only wrap raw function pointers; values that are
								// already a runtime._func struct (e.g. function
								// references or closures) must not be re-wrapped.
								if ptrT, ok := goir.AsPointerType(result.Type()); ok {
									elementT := ptrT.ElementType()
									if !elementT.IsNull() && goir.TypeIsAFunctionType(elementT) {
										result = b.createFunctionValue(ctx, result, nil, 0, location)
									}
								}
							}
						}
					}

					return result.AsValue()
				}, location)
				initializedGlobals[gv] = struct{}{}
			}

			// Increment the priority counter.
			globalPriority++
		}
	}

	// Handle embed-initialized globals.
	for obj, gv := range gvars {
		if _, ok := initializedGlobals[gv]; ok {
			continue
		}
		symbol := qualifiedName(obj.Name(), obj.Pkg())
		embedData, ok := b.config.Program.EmbedContents[symbol]
		if !ok {
			continue
		}
		location := b.location(ctx, obj.Pos())
		varType := obj.Type()

		gv.Initialize(ctx, b, globalPriority, func(ctx context.Context, b *Builder) mlir.Value {
			T := b.GetStoredType(ctx, varType)
			switch t := baseType(varType).(type) {
			case *types.Basic:
				if t.Kind() == types.String {
					return b.emitConstString(ctx, string(embedData), T, location)
				}
			case *types.Slice:
				return b.emitEmbedSlice(ctx, embedData, T, location)
			}
			panic(fmt.Sprintf("unsupported embed variable type: %s", varType))
		}, location)
		initializedGlobals[gv] = struct{}{}
		globalPriority++
	}

	// Zero initialize all other globals that are NOT externally linked.
	for obj, gv := range gvars {
		if _, ok := initializedGlobals[gv]; !ok {
			location := b.location(ctx, obj.Pos())

			// Get the symbol information.
			symbol := qualifiedName(obj.Name(), obj.Pkg())
			info := b.config.Program.Symbols.GetSymbolInfo(symbol)

			// Is this global NOT externally linked?
			if len(info.LinkName) == 0 {
				// Zero initialize the value.
				gv.Initialize(ctx, b, 0, func(ctx context.Context, b *Builder) mlir.Value {
					T := b.GetStoredType(ctx, obj.Type())
					zeroOp := goir.NewZeroOperation(b.ctx, T, location)
					appendOperation(ctx, zeroOp)
					return resultOf(zeroOp).AsValue()
				}, location)
			}
		}
	}

	// Perform a first pass on functions to create an initial state for top-level and any of their anonymous functions.
	for pkgNum, pkg := range pkgs {
		ctx = newContextWithInfo(ctx, pkg.TypesInfo)
		for _, file := range pkg.Syntax {
			for _, decl := range file.Decls {
				if decl, ok := decl.(*ast.FuncDecl); ok {
					obj := b.objectOf(ctx, decl.Name).(*types.Func)
					symbol := qualifiedFuncName(obj)
					isMain := false
					if symbol == b.config.Program.MainFunc {
						symbol = "main.main"
						isMain = true
					}

					// Is this function intrinsic?
					if isIntrinsic(symbol) {
						// Skip intrinsic functions.
						continue
					}

					symbolInfo := b.config.Program.Symbols.GetSymbolInfo(symbol)

					isPackageInit := false
					if strings.HasSuffix(symbol, ".init") && obj.Signature().Recv() == nil {
						isPackageInit = true
					}

					// Perform use analysis.
					if !b.config.DisableUseAnalysis &&
						!isPackageInit &&
						!symbolInfo.IsInterrupt &&
						!symbolInfo.ExternalLinkage &&
						!symbolInfo.Exported &&
						len(symbolInfo.LinkName) == 0 &&
						pkg.PkgPath != "runtime" &&
						!isMain {
						if _, ok := pkg.TypesInfo.Uses[decl.Name]; !ok {
							// Do not build this function.
							continue
						}
					}

					// Create the data for this function if it has NOT been encountered before or the incoming function
					// has a body and the previous did not (overrides the pre-declaration).
					existing, ok := b.funcDeclData[symbol]
					createData := !ok
					if !createData {
						if isPredeclaration(existing.decl) {
							createData = !isPredeclaration(decl)
							b.ungeneratedFuncs[symbol] = decl
						}
					}

					if createData {
						data := b.addFunctionDecl(ctx, decl)
						if data == nil {
							panic("data is nil")
						}
						if isPackageInit {
							// This is a package initializer.
							counter, ok := b.initPackageCounter[pkg]
							if !ok {
								counter = &atomic.Uint32{}
								b.initPackageCounter[pkg] = counter
							}
							data.symbol = fmt.Sprintf("%s.%d", symbol, counter.Add(1))
							data.isPackageInit = true
							data.priority = pkgNum
						}

						// Queue this function to be generated.
						queue.push(data)
					}
				}
			}
		}
	}

	// Bound the number of worker goroutines.
	g := max(1, b.config.NumWorkers)

	// Fast-path: Do nothing if there are no functions to generate.
	if len(queue.jobs) == 0 {
		return
	}

	// Begin consuming the queue.
	var activeWorkers sync.WaitGroup
	activeWorkers.Add(g)

	for c := 0; c < g; c++ {
		go func() {
			defer activeWorkers.Done()
			for {
				job := queue.pop()
				if job == nil {
					break
				}
				b.emitFunc(ctx, job)
			}
		}()
	}

	// Wait for all workers to exit.
	queue.close()
	activeWorkers.Wait()

	// Sort module-level operations by symbol name.
	symbolKeys := maps.Keys(b.addToModule)
	slices.Sort(symbolKeys)

	// Add any forward declarations whose functions should be resolved at link time.
	for linkname, op := range b.forwardDeclarations {
		if _, found := b.addToModule[linkname]; !found {
			b.appendToModule(op)
		}
	}

	// Add all module-level operations to the module's body block now.
	for _, symbol := range symbolKeys {
		b.appendToModule(b.addToModule[symbol])
	}
}

func (b *Builder) lookUpUngeneratedJob(symbol string) *ast.FuncDecl {
	b.funcDeclDataMutex.RLock()
	defer b.funcDeclDataMutex.RUnlock()
	return b.ungeneratedFuncs[symbol]
}

func (b *Builder) queueNamedTypeJobs(ctx context.Context, T *types.Named) {
	if T.TypeArgs().Len() > 0 {
		// Generic type instance — need to instantiate each method so it gets compiled.
		// Only attempt instantiation when all type args are concrete (not type parameters).
		// Inside a generic function body, the type args may still be TypeParams.
		allConcrete := true
		for i := 0; i < T.TypeArgs().Len(); i++ {
			if _, isParam := T.TypeArgs().At(i).(*types.TypeParam); isParam {
				allConcrete = false
				break
			}
		}

		if allConcrete {
			// Build the type parameter map from the instantiation's type arguments.
			typeMap := make(TypeParamMap)
			origin := T.Origin()
			for i := 0; i < origin.TypeParams().Len(); i++ {
				typeMap[origin.TypeParams().At(i).Index()] = T.TypeArgs().At(i)
			}

			// Collect the type arguments.
			targs := make([]types.Type, T.TypeArgs().Len())
			for i := range T.TypeArgs().Len() {
				targs[i] = T.TypeArgs().At(i)
			}

			// Instantiate the generic type.
			ictx := types.NewContext()
			newType, err := types.Instantiate(ictx, T, targs, true)
			if err != nil {
				panic(err)
			}
			instantiatedType := newType.(*types.Named)

			for i := 0; i < instantiatedType.NumMethods(); i++ {
				method := instantiatedType.Method(i)
				symbol := qualifiedFuncName(method)

				// Ensure the generic method declaration is processed first.
				b.queueJob(ctx, symbol)

				b.funcDeclDataMutex.RLock()
				data, ok := b.genericFuncs[symbol]
				b.funcDeclDataMutex.RUnlock()
				if ok {
					// Create (or find existing) function instance for this type instantiation.
					b.createFuncInstance(ctx, method.Type().(*types.Signature), data, typeMap)
				}
			}
			return
		}
	}

	for i := 0; i < T.NumMethods(); i++ {
		// Need to generate methods for this named type in order for interfaces to function correctly.
		symbol := qualifiedFuncName(T.Method(i))
		b.queueJob(ctx, symbol)
	}

	if _, ok := types.Unalias(T.Underlying()).(*types.Struct); ok {
		// Compute the spec-defined method set on *T (superset of T's).
		mset := types.NewMethodSet(types.NewPointer(T))
		for i := 0; i < mset.Len(); i++ {
			sel := mset.At(i)
			if len(sel.Index()) == 1 {
				// Directly declared on T (or *T) — already queued by the loop above.
				continue
			}
			b.createPromotedMethodTrampoline(ctx, T, sel)
		}
	}
}

func (b *Builder) queueJob(ctx context.Context, symbol string) {
	if decl := b.lookUpUngeneratedJob(symbol); decl != nil {
		if job := b.addFunctionDecl(ctx, decl); job != nil {
			b.addToJobQueue(ctx, job)
		}
	}
}

func (b *Builder) addToJobQueue(ctx context.Context, data *funcData) {
	if val := ctx.Value(jobQueueKey{}); val != nil {
		queue := val.(*jobQueue)
		queue.push(data)
	}
}

func (b *Builder) addFunctionDecl(ctx context.Context, decl *ast.FuncDecl) *funcData {
	b.funcDeclDataMutex.Lock()
	defer b.funcDeclDataMutex.Unlock()

	info := b.declInfo[decl]
	if info == nil {
		info = currentInfo(ctx)
	} else {
		ctx = newContextWithInfo(ctx, info)
	}

	obj := b.objectOf(ctx, decl.Name).(*types.Func)
	symbol := qualifiedFuncName(obj)
	if symbol == b.config.Program.MainFunc {
		symbol = "main.main"
	}

	signature := baseType(obj.Type()).(*types.Signature)
	symbolInfo := b.config.Program.Symbols.GetSymbolInfo(symbol)
	actualSymbol := b.resolveSymbol(symbol)

	if _, ok := b.ungeneratedFuncs[symbol]; !ok && obj.Name() != "init" {
		// Do nothing.
		return nil
	}

	data := &funcData{
		symbol:         symbol,
		linkname:       actualSymbol,
		funcType:       decl.Type,
		recv:           decl.Recv,
		body:           decl.Body,
		pos:            decl.Pos(),
		signature:      signature,
		isExported:     decl.Name.IsExported() || symbolInfo.Exported,
		isGeneric:      signature.RecvTypeParams() != nil || signature.TypeParams() != nil,
		locals:         map[types.Object]Value{},
		anonymousFuncs: map[*ast.FuncLit]*funcData{},
		instances:      []*funcData{},
		typeMap:        map[int]types.Type{},
		decl:           decl,
		info:           info,
		scope:          obj.Scope(),
		linkage:        symbolInfo.Linkage,
		attributes:     maps.Keys(symbolInfo.Attributes),
	}

	// NOTE: Have to create the function type after the func object has been initialized if the function is
	//       NOT generic.
	if !data.isGeneric {
		data.mlirType = b.GetType(ctx, obj.Type()).(goir.FunctionType)
	} else {
		b.genericFuncs[data.symbol] = data
	}

	// NOTE: Init functions do not need to be tracked as they will ONLY be called by the runtime.
	if symbol != "init" {
		b.funcDeclData[symbol] = data
	}

	// Remove this entry in the un-generated job map.
	delete(b.ungeneratedFuncs, symbol)

	return data
}

func (b *Builder) objectOf(ctx context.Context, node ast.Node) types.Object {
	info := currentInfo(ctx)
	if info == nil {
		panic("info is nil")
	}

	switch node := node.(type) {
	case *ast.Ident:
		return info.ObjectOf(node)
	case *ast.IndexExpr:
		return b.objectOf(ctx, node.X)
	case *ast.IndexListExpr:
		return b.objectOf(ctx, node.X)
	case *ast.SelectorExpr:
		if selection := info.Selections[node]; selection != nil {
			return selection.Obj()
		} else {
			return b.objectOf(ctx, node.Sel)
		}
	default:
		// Look in implicits.
		return info.Implicits[node]
	}
}

func (b *Builder) typeOf(ctx context.Context, expr ast.Expr) types.Type {
	info := currentInfo(ctx)
	if info == nil {
		panic("info is nil")
	}
	return info.TypeOf(expr)
}

func (b *Builder) hasUse(ctx context.Context, ident *ast.Ident) bool {
	info := currentInfo(ctx)
	if info == nil {
		panic("info is nil")
	}

	_, ok := info.Uses[ident]
	return ok
}

func (b *Builder) setAddr(ctx context.Context, ident *ast.Ident, addr Value) {
	if ident.Obj == nil {
		panic("no object associated with the AST node could be determined")
	}

	info := currentInfo(ctx)
	if info == nil {
		panic("info is nil")
	}

	obj := info.ObjectOf(ident)
	data := currentFuncData(ctx)
	if data != nil {
		_, ok := data.locals[obj]
		if ok {
			data.mutex.Lock()
			defer data.mutex.Unlock()
			data.locals[obj] = addr
			return
		}
	}

	b.valueCacheMutex.Lock()
	defer b.valueCacheMutex.Unlock()
	b.valueCache[obj] = addr
}

func (b *Builder) addWork(job *funcData) {
	b.work.Add(1)
	go func() {
		b.generateQueue <- job
	}()
}

func (b *Builder) addSymbol(op mlir.Operation) {
	b.symbols.Insert(op)
}

func (b *Builder) lookupSymbol(symbol string) mlir.Operation {
	return b.symbols.Lookup(symbol)
}
