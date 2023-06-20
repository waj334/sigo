package llvm

const (
	LLVMRet LLVMOpcode = iota + 1
	LLVMBr
	LLVMSwitch
	LLVMIndirectBr
	LLVMInvoke
	LLVMUnreachable
	LLVMCallBr
	LLVMFNeg
	LLVMAdd
	LLVMFAdd
	LLVMSub
	LLVMFSub
	LLVMMul
	LLVMFMul
	LLVMUDiv
	LLVMSDiv
	LLVMFDiv
	LLVMURem
	LLVMSRem
	LLVMFRem
	LLVMShl
	LLVMLShr
	LLVMAShr
	LLVMAnd
	LLVMOr
	LLVMXor
	LLVMAlloca
	LLVMLoad
	LLVMStore
	LLVMGetElementPtr
	LLVMTrunc
	LLVMZExt
	LLVMSExt
	LLVMFPToUI
	LLVMFPToSI
	LLVMUIToFP
	LLVMSIToFP
	LLVMFPTrunc
	LLVMFPExt
	LLVMPtrToInt
	LLVMIntToPtr
	LLVMBitCast
	LLVMAddrSpaceCast
	LLVMICmp
	LLVMFCmp
	LLVMPHI
	LLVMCall
	LLVMSelect
	LLVMUserOp1
	LLVMUserOp2
	LLVMVAArg
	LLVMExtractElement
	LLVMInsertElement
	LLVMShuffleVector
	LLVMExtractValue
	LLVMInsertValue
	LLVMFreeze
	LLVMFence
	LLVMAtomicCmpXchg
	LLVMAtomicRMW
	LLVMResume
	LLVMLandingPad
	LLVMCleanupRet
	LLVMCatchRet
	LLVMCatchPad
	LLVMCleanupPad
	LLVMCatchSwitch
)

const (
	VoidTypeKind LLVMTypeKind = iota
	HalfTypeKind
	FloatTypeKind
	DoubleTypeKind
	X86_FP80TypeKind
	FP128TypeKind
	PPC_FP128TypeKind
	LabelTypeKind
	IntegerTypeKind
	FunctionTypeKind
	StructTypeKind
	ArrayTypeKind
	PointerTypeKind
	VectorTypeKind
	MetadataTypeKind
	X86_MMXTypeKind
	TokenTypeKind
	ScalableVectorTypeKind
	BFloatTypeKind
	X86_AMXTypeKind
	TargetExtTypeKind
)

const (
	ExternalLinkage LLVMLinkage = iota // Externally visible function.
	AvailableExternallyLinkage
	LinkOnceAnyLinkage         // Keep one copy of function when linking (inline)
	LinkOnceODRLinkage         // Same, but only replaced by something equivalent.
	LinkOnceODRAutoHideLinkage // Obsolete.
	WeakAnyLinkage             // Keep one copy of function when linking (weak)
	WeakODRLinkage             // Same, but only replaced by something equivalent.
	AppendingLinkage           // Special purpose, only applies to global arrays.
	InternalLinkage            // Rename collisions when linking (static functions)
	PrivateLinkage             // Like Internal, but omit from symbol table.
	DLLImportLinkage           // Obsolete.
	DLLExportLinkage           // Obsolete.
	ExternalWeakLinkage        // ExternalWeak linkage description.
	GhostLinkage               // Obsolete.
	CommonLinkage              // Tentative definitions.
	LinkerPrivateLinkage       // Like Private, but linker removes.
	LinkerPrivateWeakLinkage   // Like LinkerPrivate, but is weak.
)

const (
	DefaultVisibility   LLVMVisibility = iota // The GV is visible.
	HiddenVisibility                          // The GV is hidden.
	ProtectedVisibility                       // The GV is protected.
)

const (
	NoUnnamedAddr     LLVMUnnamedAddr = iota // Address of the GV is significant.
	LocalUnnamedAddr                         // Address of the GV is locally insignificant.
	GlobalUnnamedAddr                        // Address of the GV is globally insignificant.
)

const (
	DefaultStorageClass   LLVMDLLStorageClass = iota
	DLLImportStorageClass                     // Function to be imported from DLL.
	DLLExportStorageClass                     // Function to be accessible from DLL.
)

type LLVMCallConv uint

const (
	CCallConv    LLVMCallConv = iota
	FastCallConv LLVMCallConv = iota + 7
	ColdCallConv
	GHCCallConv
	HiPECallConv
	WebKitJSCallConv
	AnyRegCallConv
	PreserveMostCallConv
	PreserveAllCallConv
	SwiftCallConv
	CXXFASTTLSCallConv
	X86StdcallCallConv LLVMCallConv = iota + 53
	X86FastcallCallConv
	ARMAPCSCallConv
	ARMAAPCSCallConv
	ARMAAPCSVFPCallConv
	MSP430INTRCallConv
	X86ThisCallCallConv
	PTXKernelCallConv
	PTXDeviceCallConv
	SPIRFUNCCallConv LLVMCallConv = iota + 55
	SPIRKERNELCallConv
	IntelOCLBICallConv
	X8664SysVCallConv
	Win64CallConv
	X86VectorCallCallConv
	HHVMCallConv
	HHVMCCallConv
	X86INTRCallConv
	AVRINTRCallConv
	AVRSIGNALCallConv
	AVRBUILTINCallConv
	AMDGPUVSCallConv
	AMDGPUGSCallConv
	AMDGPUPSCallConv
	AMDGPUCSCallConv
	AMDGPUKERNELCallConv
	X86RegCallCallConv
	AMDGPUHSCallConv
	MSP430BUILTINCallConv
	AMDGPULSCallConv
	AMDGPUESCallConv
)

const (
	ArgumentValueKind LLVMValueKind = iota
	BasicBlockValueKind
	MemoryUseValueKind
	MemoryDefValueKind
	MemoryPhiValueKind
	FunctionValueKind
	GlobalAliasValueKind
	GlobalIFuncValueKind
	GlobalVariableValueKind
	BlockAddressValueKind
	ConstantExprValueKind
	ConstantArrayValueKind
	ConstantStructValueKind
	ConstantVectorValueKind
	UndefValueValueKind
	ConstantAggregateZeroValueKind
	ConstantDataArrayValueKind
	ConstantDataVectorValueKind
	ConstantIntValueKind
	ConstantFPValueKind
	ConstantPointerNullValueKind
	ConstantTokenNoneValueKind
	MetadataAsValueValueKind
	InlineAsmValueKind
	InstructionValueKind
	PoisonValueValueKind
	ConstantTargetNoneValueKind
)

const (
	IntEQ  LLVMIntPredicate = iota + 32 // equal
	IntNE                               // not equal
	IntUGT                              // unsigned greater than
	IntUGE                              // unsigned greater or equal
	IntULT                              // unsigned less than
	IntULE                              // unsigned less or equal
	IntSGT                              // signed greater than
	IntSGE                              // signed greater or equal
	IntSLT                              // signed less than
	IntSLE                              // signed less or equal
)

const (
	RealPredicateFalse = iota // Always false (always folded)
	RealOEQ                   // True if ordered and equal
	RealOGT                   // True if ordered and greater than
	RealOGE                   // True if ordered and greater than or equal
	RealOLT                   // True if ordered and less than
	RealOLE                   // True if ordered and less than or equal
	RealONE                   // True if ordered and operands are unequal
	RealORD                   // True if ordered (no nans)
	RealUNO                   // True if unordered: isnan(X) | isnan(Y)
	RealUEQ                   // True if unordered or equal
	RealUGT                   // True if unordered or greater than
	RealUGE                   // True if unordered, greater than, or equal
	RealULT                   // True if unordered or less than
	RealULE                   // True if unordered, less than, or equal
	RealUNE                   // True if unordered or not equal
	RealPredicateTrue         // Always true (always folded)
)

type LLVMLandingPadClauseTy uint

const (
	LandingPadCatch LLVMLandingPadClauseTy = iota
	LandingPadFilter
)

const (
	NotThreadLocal LLVMThreadLocalMode = iota
	GeneralDynamicTLSModel
	LocalDynamicTLSModel
	InitialExecTLSModel
	LocalExecTLSModel
)

const (
	AtomicOrderingNotAtomic              LLVMAtomicOrdering = iota     // A load or store which is not atomic.
	AtomicOrderingUnordered                                            // Lowest level of atomicity, guarantees somewhat sane results, lock free.
	AtomicOrderingMonotonic                                            // guarantees that if you take all the operations affecting a specific address, a consistent ordering exists
	AtomicOrderingAcquire                LLVMAtomicOrdering = iota + 1 // Acquire provides a barrier of the sort necessary to acquire a lock to access other memory with normal loads and stores.
	AtomicOrderingRelease                                              // Release is similar to Acquire, but with a barrier of the sort necessary to release a lock.
	AtomicOrderingAcquireRelease                                       // provides both an Acquire and a Release barrier (for fences and operations which both read and write memory).
	AtomicOrderingSequentiallyConsistent                               // provides Acquire semantics for loads and Release semantics for stores. Additionally, it guarantees that a total ordering exists between all SequentiallyConsistent operations.
)

const (
	AtomicRMWBinOpXchg LLVMAtomicRMWBinOp = iota // Set the new value and return the one old.
	AtomicRMWBinOpAdd                            // Add a value and return the old one.
	AtomicRMWBinOpSub                            // Subtract a value and return the old one.
	AtomicRMWBinOpAnd                            // And a value and return the old one.
	AtomicRMWBinOpNand                           // Not-And a value and return the old one.
	AtomicRMWBinOpOr                             // OR a value and return the old one.
	AtomicRMWBinOpXor                            // Xor a value and return the old one.
	AtomicRMWBinOpMax                            // Sets the value if it's greater than the original using a signed comparison and return the old one.
	AtomicRMWBinOpMin                            // Sets the value if it's Smaller than the original using a signed comparison and return the old one.
	AtomicRMWBinOpUMax                           // Sets the value if it's greater than the original using an unsigned comparison and return the old one.
	AtomicRMWBinOpUMin                           // Sets the value if it's greater than the original using an unsigned comparison and return the old one.
	AtomicRMWBinOpFAdd                           // Add a floating point value and return the old one.
	AtomicRMWBinOpFSub                           // Subtract a floating point value and return the old one.
	AtomicRMWBinOpFMax                           // Sets the value if it's greater than the original using an floating point comparison and return the old one.
	AtomicRMWBinOpFMin                           // Sets the value if it's smaller than the original using an floating point comparison and return the old one.
)

const (
	DSError LLVMDiagnosticSeverity = iota
	DSWarning
	DSRemark
	DSNote
)

const (
	InlineAsmDialectATT LLVMInlineAsmDialect = iota
	InlineAsmDialectIntel
)

const (
	ModuleFlagBehaviorError        LLVMModuleFlagBehavior = iota // Emits an error if two values disagree, otherwise the resulting value is that of the operands.
	ModuleFlagBehaviorWarning                                    // Emits a warning if two values disagree. The result value will be the operand for the flag from the first module being linked.
	ModuleFlagBehaviorRequire                                    // Adds a requirement that another module flag be present and have a specified value after linking is performed.
	ModuleFlagBehaviorOverride                                   // Uses the specified value, regardless of the behavior or value of the other module.
	ModuleFlagBehaviorAppend                                     // Appends the two values, which are required to be metadata nodes.
	ModuleFlagBehaviorAppendUnique                               // Appends the two values, which are required to be metadata nodes.
)
