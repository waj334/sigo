package ssa

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"golang.org/x/exp/maps"
	"golang.org/x/exp/slices"
	"golang.org/x/tools/go/packages"
	"omibyte.io/sigo/llvm"
	"omibyte.io/sigo/mlir"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
)

type Config struct {
	Fset               *token.FileSet
	Info               *types.Info
	Ctx                mlir.Context
	Module             mlir.Module
	Sizes              *types.StdSizes
	Program            *Program
	NumWorkers         int
	DisableUseAnalysis bool
}

type Builder struct {
	ctx mlir.Context

	// Thread-safe structures (No async writes):
	config        Config
	program       *Program
	declaredTypes map[types.Type]struct{}
	genericFuncs  map[string]*funcData
	diFiles       map[*token.File]mlir.Attribute
	compileUnits  map[*token.File]mlir.Attribute
	symbols       mlir.SymbolTable

	// Thread-unsafe structures:
	funcDeclDataMutex sync.RWMutex
	funcDeclData      map[string]*funcData
	ungeneratedFuncs  map[string]*ast.FuncDecl

	typeCache      map[types.Type]mlir.Type
	typeCacheMutex sync.RWMutex

	valueCache      map[types.Object]Value
	valueCacheMutex sync.RWMutex

	addToModuleMutex sync.Mutex
	addToModule      map[string]mlir.Operation

	thunkMutex sync.Mutex
	thunks     map[string]struct{}

	// Misc:
	generateQueue      chan *funcData
	work               atomic.Int32
	generateQueueMutex sync.Mutex

	i1    mlir.Type
	si    mlir.Type
	si8   mlir.Type
	si32  mlir.Type
	si64  mlir.Type
	ui    mlir.Type
	ui8   mlir.Type
	ui32  mlir.Type
	ui64  mlir.Type
	f32   mlir.Type
	f64   mlir.Type
	c64   mlir.Type
	c128  mlir.Type
	ptr   mlir.Type
	uiptr mlir.Type
	str   mlir.Type

	typeInfoPtr mlir.Type

	_chan      mlir.Type
	_map       mlir.Type
	_slice     mlir.Type
	_string    mlir.Type
	_interface mlir.Type
	_any       mlir.Type
	_func      mlir.Type

	anyType *types.Interface

	_noLoc mlir.Location

	initPackageCounter map[*packages.Package]*atomic.Uint32
}

type jobQueue struct {
	jobs []*funcData
}

func NewBuilder(config Config) *Builder {
	builder := &Builder{
		config:        config,
		typeCache:     map[types.Type]mlir.Type{},
		valueCache:    map[types.Object]Value{},
		ctx:           config.Ctx,
		program:       config.Program,
		symbols:       mlir.SymbolTableCreate(mlir.ModuleGetOperation(config.Module)),
		generateQueue: make(chan *funcData),
		thunks:        map[string]struct{}{},
		genericFuncs:  map[string]*funcData{},

		declaredTypes: map[types.Type]struct{}{},

		ungeneratedFuncs: map[string]*ast.FuncDecl{},
		funcDeclData:     map[string]*funcData{},
		addToModule:      map[string]mlir.Operation{},

		diFiles:            map[*token.File]mlir.Attribute{},
		compileUnits:       map[*token.File]mlir.Attribute{},
		initPackageCounter: map[*packages.Package]*atomic.Uint32{},
	}

	// Create all basic types up front for ease of use later.
	builder.i1 = builder.GetType(context.Background(), types.Typ[types.Bool])
	builder.si = builder.GetType(context.Background(), types.Typ[types.Int])
	builder.si8 = builder.GetType(context.Background(), types.Typ[types.Int8])
	builder.si32 = builder.GetType(context.Background(), types.Typ[types.Int32])
	builder.si64 = builder.GetType(context.Background(), types.Typ[types.Int64])
	builder.ui = builder.GetType(context.Background(), types.Typ[types.Uint])
	builder.ui8 = builder.GetType(context.Background(), types.Typ[types.Uint8])
	builder.ui32 = builder.GetType(context.Background(), types.Typ[types.Uint32])
	builder.ui64 = builder.GetType(context.Background(), types.Typ[types.Uint64])
	builder.f32 = builder.GetType(context.Background(), types.Typ[types.Float32])
	builder.f64 = builder.GetType(context.Background(), types.Typ[types.Float64])
	builder.c64 = builder.GetType(context.Background(), types.Typ[types.Complex64])
	builder.c128 = builder.GetType(context.Background(), types.Typ[types.Complex128])
	builder.ptr = builder.GetType(context.Background(), types.Typ[types.UnsafePointer])
	builder.uiptr = builder.GetType(context.Background(), types.Typ[types.Uintptr])
	builder.str = builder.GetType(context.Background(), types.Typ[types.String])

	builder.typeInfoPtr = builder.GetType(context.Background(), types.NewPointer(config.Program.LookupType("runtime", "_type")))

	builder._chan = builder.GetType(context.Background(), config.Program.LookupType("runtime", "_channel"))
	builder._map = builder.GetType(context.Background(), config.Program.LookupType("runtime", "_map"))
	builder._slice = builder.GetType(context.Background(), config.Program.LookupType("runtime", "_slice"))
	builder._string = builder.GetType(context.Background(), config.Program.LookupType("runtime", "_string"))
	builder._interface = builder.GetType(context.Background(), config.Program.LookupType("runtime", "_interface"))

	builder.anyType = types.NewInterfaceType(nil, nil).Complete()
	builder._any = builder.GetType(context.Background(), builder.anyType)

	// {func_ptr, env_ptr}
	//builder._func = mlir.GoCreateNamedType(mlir.GoCreateLiteralStructType(config.Ctx, []mlir.Type{builder.ptr, builder.ptr}), "func")
	builder._func = builder.GetType(context.Background(), config.Program.LookupType("runtime", "_func"))

	builder._noLoc = builder.location(0)

	// Bind dialect types to runtime types.
	typeMap := map[string]string{
		"chan":                "runtime._channel",
		"interface":           "runtime._interface",
		"map":                 "runtime._map",
		"slice":               "runtime._slice",
		"string":              "runtime._string",
		"type":                "runtime._type",
		"namedTypeData":       "runtime._namedTypeData",
		"funcData":            "runtime._funcData",
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
		mlir.GoBindRuntimeType(config.Module, k, builder.GetType(context.Background(), T))
	}

	return builder
}

func (b *Builder) GeneratePackages(ctx context.Context, pkgs []*packages.Package) {
	// All operations should go to the module body by default.
	ctx = newContextWithCurrentBlock(ctx)
	moduleRegion := mlir.OperationGetFirstRegion(mlir.ModuleGetOperation(b.config.Module))
	ctx = newContextWithRegion(ctx, moduleRegion)
	moduleBlock := mlir.ModuleGetBody(b.config.Module)
	setCurrentBlock(ctx, moduleBlock)

	// Create debug information for each file.
	producerAttr := mlir.StringAttrGet(b.ctx, "SiGo")
	b.config.Fset.Iterate(func(file *token.File) bool {
		fname := file.Name()
		if evalPath, err := filepath.EvalSymlinks(fname); err == nil {
			fname = evalPath
		}

		nameAttr := mlir.StringAttrGet(b.ctx, filepath.Base(file.Name()))
		fnameAttr := mlir.StringAttrGet(b.ctx, fname)
		diFileAttr := mlir.LLVMDIFileAttrGet(b.ctx, nameAttr, fnameAttr)
		b.diFiles[file] = diFileAttr

		// Create a matching compile unit for this file.
		idAttr := mlir.DistinctAttrGet(fnameAttr)
		compileUnitAttr := mlir.LLVMDICompileUnitAttrGet(
			b.ctx,
			idAttr,
			uint(llvm.DWARFSourceLanguageGo),
			diFileAttr,
			producerAttr,
			false,
			mlir.LLVMDIEmissionKindFull,
			mlir.LLVMDINameTableKindDefault,
		)
		b.compileUnits[file] = compileUnitAttr
		return true
	})

	// Create the initial work queue.
	initialWork := &jobQueue{
		jobs: make([]*funcData, 0, 1000),
	}
	ctx = context.WithValue(ctx, jobQueueKey{}, initialWork)

	// Populate un-generated functions list first so global initializers can add work to the initial job queue.
	for _, pkg := range pkgs {
		for _, file := range pkg.Syntax {
			for _, decl := range file.Decls {
				if decl, ok := decl.(*ast.FuncDecl); ok {
					obj := b.objectOf(decl.Name).(*types.Func)
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
					actualSymbol := b.resolveSymbol(symbol)
					b.ungeneratedFuncs[actualSymbol] = decl
				}
			}
		}
	}

	// Emit constants and global variables serially.
	gvars := map[types.Object]*GlobalValue{}
	for _, pkg := range pkgs {
		for _, file := range pkg.Syntax {
			for _, decl := range file.Decls {
				switch decl := decl.(type) {
				case *ast.GenDecl:
					switch decl.Tok {
					case token.TYPE:
						for _, spec := range decl.Specs {
							spec := spec.(*ast.TypeSpec)
							obj := b.objectOf(spec.Name)
							b.createTypeDeclaration(ctx, obj.Type(), obj.Pos())
						}
					case token.VAR:
						for _, spec := range decl.Specs {
							spec := spec.(*ast.ValueSpec)
							for _, ident := range spec.Names {
								gvar := b.emitGlobalVar(ctx, ident)
								gvars[b.objectOf(ident)] = gvar
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
				gv := gvars[lhs]
				location := b.location(lhs.Pos())

				// Initialize this global.
				gv.Initialize(ctx, b, globalPriority, func(ctx context.Context, builder *Builder) mlir.Value {
					// Nil types will be untyped, so set up type inference.
					ctx = newContextWithLhsList(ctx, []types.Type{lhs.Type()})
					ctx = newContextWithRhsIndex(ctx, 0)
					rhsType := b.typeOf(initializer.Rhs)

					result := b.emitExpr(ctx, initializer.Rhs)[0]

					if rhsType != nil {
						if T, ok := rhsType.(*types.Named); ok {
							b.queueNamedTypeJobs(ctx, T)
						}

						switch baseType(lhs.Type()).(type) {
						case *types.Interface:
							if !types.Identical(lhs.Type(), rhsType) {
								switch rhsType.(type) {
								case *types.Interface:
									result = b.emitChangeType(ctx, lhs.Type(), result, location)
								default:
									result = b.emitInterfaceValue(ctx, lhs.Type(), result, location)
								}
							}
						default:
							switch b.typeOf(initializer.Rhs).(type) {
							case *types.Signature:
								result = b.createFunctionValue(ctx, result, nil, location)
							}
						}
					}

					return result
				}, location)
				initializedGlobals[gv] = struct{}{}
			}

			// Increment the priority counter.
			globalPriority++
		}
	}

	// Zero initialize all other globals that are NOT externally linked.
	for obj, gv := range gvars {
		if _, ok := initializedGlobals[gv]; !ok {
			location := b.location(obj.Pos())

			// Get the symbol information.
			symbol := qualifiedName(obj.Name(), obj.Pkg())
			info := b.config.Program.Symbols.GetSymbolInfo(symbol)

			// Is this global NOT externally linked?
			if len(info.LinkName) == 0 {
				// Zero initialize the value.
				gv.Initialize(ctx, b, 0, func(ctx context.Context, b *Builder) mlir.Value {
					T := b.GetStoredType(ctx, obj.Type())
					zeroOp := mlir.GoCreateZeroOperation(b.ctx, T, location)
					appendOperation(ctx, zeroOp)
					return resultOf(zeroOp)
				}, location)
			}
		}
	}

	// Perform a first pass on functions to create an initial state for top-level and any of their anonymous functions.
	for pkgNum, pkg := range pkgs {
		for _, file := range pkg.Syntax {
			for _, decl := range file.Decls {
				if decl, ok := decl.(*ast.FuncDecl); ok {
					obj := b.objectOf(decl.Name).(*types.Func)
					symbol := qualifiedFuncName(obj)
					isMain := false
					if symbol == b.config.Program.MainFunc {
						symbol = "main.main"
						isMain = true
					}

					// Is this function an intrinsic?
					if isIntrinsic(symbol) {
						// Skip intrinsic functions.
						continue
					}

					symbolInfo := b.config.Program.Symbols.GetSymbolInfo(symbol)
					actualSymbol := b.resolveSymbol(symbol)

					// Mark this function as un-generated.
					//b.ungeneratedFuncs[actualSymbol] = decl

					isPackageInit := false
					if strings.HasSuffix(actualSymbol, ".init") {
						isPackageInit = true
					}

					// Perform use analysis.
					if !b.config.DisableUseAnalysis &&
						!isPackageInit &&
						!symbolInfo.IsInterrupt &&
						!symbolInfo.ExternalLinkage &&
						len(symbolInfo.LinkName) == 0 &&
						pkg.PkgPath != "runtime" &&
						!isMain {
						if _, ok := b.config.Info.Uses[decl.Name]; !ok {
							// Do not build this function.
							continue
						}
					}

					// Create the data for this function if it has NOT been encountered before or the incoming function
					// has a body and the previous did not (overrides the pre-declaration).
					existing, ok := b.funcDeclData[actualSymbol]
					createData := !ok
					if !createData {
						if isPredeclaration(existing.decl) {
							createData = !isPredeclaration(decl)
							b.ungeneratedFuncs[actualSymbol] = decl
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
							data.symbol = fmt.Sprintf("%s.%d", actualSymbol, counter.Add(1))
							data.isPackageInit = true
							data.priority = pkgNum
						}

						// Queue this function to be generated.
						initialWork.jobs = append(initialWork.jobs, data)
					}
				}
			}
		}
	}

	// Bound the number of worker goroutines.
	g := max(1, b.config.NumWorkers)

	// Fast-path: Do nothing if there are no functions to generate.
	if len(initialWork.jobs) == 0 {
		return
	}

	generateQueue := make(chan *funcData)

	// Track the number of pending jobs.
	//var pendingJobs atomic.Int32
	//pendingJobs.Store(int32(len(initialWork.jobs)))
	var pendingJobs sync.WaitGroup
	pendingJobs.Add(len(initialWork.jobs))

	// Define a function to queue new jobs.
	queueFunc := func(queue *jobQueue) {
		// Queue each job.
		for _, job := range queue.jobs {
			generateQueue <- job
		}
	}

	// Queue all initial jobs.
	// NOTE: This should guarantee each function has a chance to queue new jobs, causing the job queue channel to be
	//       closed later.
	go queueFunc(initialWork)

	// Begin consuming the queue.
	var activeWorkers sync.WaitGroup
	activeWorkers.Add(g)
	for c := 0; c < g; c++ {
		go func() {
			defer activeWorkers.Done()
			for job := range generateQueue {
				// Create a new job queue for when functions need other functions to be generated.
				queue := &jobQueue{jobs: make([]*funcData, 0, 1000)}
				ctx = context.WithValue(ctx, jobQueueKey{}, queue)
				b.emitFunc(ctx, job)

				if len(queue.jobs) > 0 {
					// Add to the work counter.
					pendingJobs.Add(len(queue.jobs))

					// Queue new jobs requested by the last job.
					go queueFunc(queue)
				}

				// Finally, signal that this job is done.
				pendingJobs.Done()
			}
		}()
	}

	// Wait for queuing to be completed.
	go func() {
		// Wait for all pending jobs to be done.
		pendingJobs.Wait()

		// Close the queue channel so that the worker goroutines can exit.
		close(generateQueue)
	}()

	// Wait for all workers to exit.
	activeWorkers.Wait()

	// Sort module-level operations by symbol name.
	symbolKeys := maps.Keys(b.addToModule)
	slices.Sort(symbolKeys)

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
	for i := 0; i < T.NumMethods(); i++ {
		// Need to generate methods for this named type in order for interfaces to function correctly.
		symbol := qualifiedFuncName(T.Method(i))
		b.queueJob(ctx, symbol)
	}
}

func (b *Builder) queueJob(ctx context.Context, symbol string) {
	if val := ctx.Value(jobQueueKey{}); val != nil {
		if decl := b.lookUpUngeneratedJob(symbol); decl != nil {
			if job := b.addFunctionDecl(ctx, decl); job != nil {
				queue := val.(*jobQueue)
				queue.jobs = append(queue.jobs, job)
			}
		}
	}
}

func (b *Builder) addFunctionDecl(ctx context.Context, decl *ast.FuncDecl) *funcData {
	b.funcDeclDataMutex.Lock()
	defer b.funcDeclDataMutex.Unlock()

	obj := b.objectOf(decl.Name).(*types.Func)
	symbol := qualifiedFuncName(obj)
	if symbol == b.config.Program.MainFunc {
		symbol = "main.main"
	}

	signature := obj.Type().Underlying().(*types.Signature)
	symbolInfo := b.config.Program.Symbols.GetSymbolInfo(symbol)
	actualSymbol := b.resolveSymbol(symbol)

	if _, ok := b.ungeneratedFuncs[actualSymbol]; !ok && obj.Name() != "init" {
		// Do nothing.
		return nil
	}

	data := &funcData{
		symbol:         actualSymbol,
		funcType:       decl.Type,
		recv:           decl.Recv,
		body:           decl.Body,
		pos:            decl.Pos(),
		signature:      signature,
		isExported:     decl.Name.IsExported() || symbolInfo.Exported,
		isGeneric:      signature.RecvTypeParams() != nil || signature.TypeParams() != nil,
		locals:         map[string]Value{},
		anonymousFuncs: map[*ast.FuncLit]*funcData{},
		instances:      map[*types.Signature]*funcData{},
		typeMap:        map[int]types.Type{},
		decl:           decl,
	}

	// NOTE: Have to create the function type after the func object has been initialized if the function is
	//       NOT generic.
	if !data.isGeneric {
		data.mlirType = b.GetType(ctx, obj.Type())
	} else {
		b.genericFuncs[data.symbol] = data
	}

	// NOTE: Init functions do not need to be tracked as they will ONLY be called by the runtime.
	if symbol != "init" {
		b.funcDeclData[actualSymbol] = data
	}

	// Remove this entry in the un-generated job map.
	delete(b.ungeneratedFuncs, actualSymbol)

	return data
}

func (b *Builder) objectOf(node ast.Node) types.Object {
	switch node := node.(type) {
	case *ast.Ident:
		return b.config.Info.ObjectOf(node)
	case *ast.SelectorExpr:
		if selection := b.config.Info.Selections[node]; selection != nil {
			return selection.Obj()
		} else {
			return b.objectOf(node.Sel)
		}
	default:
		// Look in implicits.
		return b.config.Info.Implicits[node]
	}
}

func (b *Builder) typeOf(expr ast.Expr) types.Type {
	return b.config.Info.TypeOf(expr)
}

func (b *Builder) hasUse(ident *ast.Ident) bool {
	_, ok := b.config.Info.Uses[ident]
	return ok
}

func (b *Builder) setAddr(ident *ast.Ident, addr Value) {
	if ident.Obj == nil {
		panic("no object associated with the AST node could be determined")
	}

	obj := b.config.Info.ObjectOf(ident)

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
	mlir.SymbolTableInsert(b.symbols, op)
}

func (b *Builder) lookupSymbol(symbol string) mlir.Operation {
	op := mlir.SymbolTableLookup(b.symbols, symbol)
	if mlir.OperationIsNull(op) {
		return nil
	}
	return op
}
