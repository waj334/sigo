%module llvm

%include inttypes.i
%include typemaps.i

%header %{
#include <stdbool.h>
#include "llvm-c/Analysis.h"
#include "llvm-c/BitReader.h"
#include "llvm-c/BitWriter.h"
#include "llvm-c/blake3.h"
#include "llvm-c/Comdat.h"
#include "llvm-c/Core.h"
#include "llvm-c/DataTypes.h"
#include "llvm-c/DebugInfo.h"
#include "llvm-c/Deprecated.h"
#include "llvm-c/Disassembler.h"
#include "llvm-c/DisassemblerTypes.h"
#include "llvm-c/Error.h"
#include "llvm-c/ErrorHandling.h"
//#include "llvm-c/ExecutionEngine.h"
#include "llvm-c/ExternC.h"
#include "llvm-c/Initialization.h"
#include "llvm-c/IRReader.h"
#include "llvm-c/Linker.h"
//#include "llvm-c/LLJIT.h"
//#include "llvm-c/lto.h"
#include "llvm-c/Object.h"
//#include "llvm-c/Orc.h"
//#include "llvm-c/OrcEE.h"
//#include "llvm-c/Remarks.h"
#include "llvm-c/Support.h"
#include "llvm-c/Target.h"
#include "llvm-c/TargetMachine.h"
#include "llvm-c/Types.h"
#include "llvm-c/Transforms/InstCombine.h"
#include "llvm-c/Transforms/IPO.h"
#include "llvm-c/Transforms/PassBuilder.h"
#include "llvm-c/Transforms/PassManagerBuilder.h"
#include "llvm-c/Transforms/Scalar.h"
#include "llvm-c/Transforms/Utils.h"
#include "llvm-c/Transforms/Vectorize.h"
%}

%insert(cgo_comment_typedefs) %{
#include <stdlib.h>
#include "llvm-c/Analysis.h"
#include "llvm-c/BitReader.h"
#include "llvm-c/BitWriter.h"
#include "llvm-c/blake3.h"
#include "llvm-c/Comdat.h"
#include "llvm-c/Core.h"
#include "llvm-c/DataTypes.h"
#include "llvm-c/DebugInfo.h"
#include "llvm-c/Deprecated.h"
#include "llvm-c/Disassembler.h"
#include "llvm-c/DisassemblerTypes.h"
#include "llvm-c/Error.h"
#include "llvm-c/ErrorHandling.h"
//#include "llvm-c/ExecutionEngine.h"
#include "llvm-c/ExternC.h"
#include "llvm-c/Initialization.h"
#include "llvm-c/IRReader.h"
#include "llvm-c/Linker.h"
//#include "llvm-c/LLJIT.h"
//#include "llvm-c/lto.h"
#include "llvm-c/Object.h"
//#include "llvm-c/Orc.h"
//#include "llvm-c/OrcEE.h"
//#include "llvm-c/Remarks.h"
#include "llvm-c/Support.h"
#include "llvm-c/Target.h"
#include "llvm-c/TargetMachine.h"
#include "llvm-c/Types.h"
#include "llvm-c/Transforms/InstCombine.h"
#include "llvm-c/Transforms/IPO.h"
#include "llvm-c/Transforms/PassBuilder.h"
#include "llvm-c/Transforms/PassManagerBuilder.h"
#include "llvm-c/Transforms/Scalar.h"
#include "llvm-c/Transforms/Utils.h"
#include "llvm-c/Transforms/Vectorize.h"
%}

%typemap(gotype) LLVMTypeKind "LLVMTypeKind"
%typemap(gotype) LLVMIntPredicate "LLVMIntPredicate"
%typemap(gotype) LLVMRealPredicate "LLVMRealPredicate"

%{
typedef int                                     LLVMBool;
typedef unsigned                                LLVMDWARFTypeEncoding;
%}

%ignore LLVMTypeKind;
%ignore LLVMIntPredicate;
%ignore LLVMRealPredicate;

// Apply bool typemaps to LLVMBool
%apply bool { LLVMBool };

// Replace C arrays with slices

// *********** Int
%typemap(gotype)    (unsigned *IntParams, unsigned IntParamCount) "[]uint";
%typemap(goin)      (unsigned *IntParams, unsigned IntParamCount)
{
    $result = $input
}

// *********** Uint64
%typemap(in)        (unsigned *IntParams, unsigned IntParamCount){}

%typemap(gotype)    (uint64_t *Addr, size_t Length) "[]uint64";
%typemap(goin)      (uint64_t *Addr, size_t Length)
{
    $result = $input
}

%typemap(in)        (uint64_t *Addr, size_t Length){}

// *********** Strings
// String inputs expecting length
%typemap(in)        (const char *Name, size_t NameLen),
                    (const char *Filename, size_t FilenameLen),
                    (const char *Directory, size_t DirectoryLen),
                    (const char *UniqueId, size_t UniqueIdLen),
                    (const char *LinkageName, size_t LinkageNameLen),
                    (const char *Flags, size_t FlagsLen),
                    (const char *Producer, size_t ProducerLen),
                    (const char *SplitName, size_t SplitNameLen),
                    (const char *SysRoot, size_t SysRootLen),
                    (const char *SDK, size_t SDKLen),
                    (const char *Str, unsigned Length),
                    (const char *UniqueIdentifier, size_t UniqueIdentifierLen),
                    (const char *Str, unsigned SLen),
                    (const char *Str, size_t SLen),
                    (const char *Name, unsigned SLen),
                    (const char *Name, size_t SLen),
                    (const char *Linkage, size_t LinkLen)

{
    $1 = $input;
    $2 = strlen($input);
}

%typemap(out)       (const char *Name, size_t NameLen),
                    (const char *Filename, size_t FilenameLen),
                    (const char *Directory, size_t DirectoryLen),
                    (const char *UniqueId, size_t UniqueIdLen),
                    (const char *LinkageName, size_t LinkageNameLen),
                    (const char *Flags, size_t FlagsLen),
                    (const char *Producer, size_t ProducerLen),
                    (const char *SplitName, size_t SplitNameLen),
                    (const char *SysRoot, size_t SysRootLen),
                    (const char *SDK, size_t SDKLen),
                    (const char *Str, unsigned Length),
                    (const char *UniqueIdentifier, size_t UniqueIdentifierLen),
                    (const char *Str, unsigned SLen),
                    (const char *Str, size_t SLen),
                    (const char *Name, unsigned SLen),
                    (const char *Name, size_t SLen),
                    (const char *Linkage, size_t LinkLen)
{
    $result = $1;
}

%typemap(freearg)   (const char *Name, size_t NameLen),
                    (const char *Filename, size_t FilenameLen),
                    (const char *Directory, size_t DirectoryLen),
                    (const char *UniqueId, size_t UniqueIdLen),
                    (const char *LinkageName, size_t LinkageNameLen),
                    (const char *Flags, size_t FlagsLen),
                    (const char *Producer, size_t ProducerLen),
                    (const char *SplitName, size_t SplitNameLen),
                    (const char *SysRoot, size_t SysRootLen),
                    (const char *SDK, size_t SDKLen),
                    (const char *Str, unsigned Length),
                    (const char *UniqueIdentifier, size_t UniqueIdentifierLen),
                    (const char *Str, unsigned SLen),
                    (const char *Str, size_t SLen),
                    (const char *Name, unsigned SLen),
                    (const char *Name, size_t SLen),
                    (const char *Linkage, size_t LinkLen)
{
    free($input);
}

%typemap(imtype)    (const char *Name, size_t NameLen),
                    (const char *Filename, size_t FilenameLen),
                    (const char *Directory, size_t DirectoryLen),
                    (const char *UniqueId, size_t UniqueIdLen),
                    (const char *LinkageName, size_t LinkageNameLen),
                    (const char *Flags, size_t FlagsLen),
                    (const char *Producer, size_t ProducerLen),
                    (const char *SplitName, size_t SplitNameLen),
                    (const char *SysRoot, size_t SysRootLen),
                    (const char *SDK, size_t SDKLen),
                    (const char *Str, unsigned Length),
                    (const char *UniqueIdentifier, size_t UniqueIdentifierLen),
                    (const char *Str, unsigned SLen),
                    (const char *Str, size_t SLen),
                    (const char *Name, unsigned SLen),
                    (const char *Name, size_t SLen),
                    (const char *Linkage, size_t LinkLen)
                    "*C.char";

%typemap(gotype)    (const char *Name, size_t NameLen),
                    (const char *Filename, size_t FilenameLen),
                    (const char *Directory, size_t DirectoryLen),
                    (const char *UniqueId, size_t UniqueIdLen),
                    (const char *LinkageName, size_t LinkageNameLen),
                    (const char *Flags, size_t FlagsLen),
                    (const char *Producer, size_t ProducerLen)
                    (const char *SplitName, size_t SplitNameLen),
                    (const char *SysRoot, size_t SysRootLen),
                    (const char *SDK, size_t SDKLen),
                    (const char *Str, unsigned Length),
                    (const char *UniqueIdentifier, size_t UniqueIdentifierLen),
                    (const char *Str, unsigned SLen),
                    (const char *Str, size_t SLen),
                    (const char *Name, unsigned SLen),
                    (const char *Name, size_t SLen),
                    (const char *Linkage, size_t LinkLen)
                    "string";

%typemap(goin)      (const char *Name, size_t NameLen),
                    (const char *Filename, size_t FilenameLen),
                    (const char *Directory, size_t DirectoryLen),
                    (const char *UniqueId, size_t UniqueIdLen),
                    (const char *LinkageName, size_t LinkageNameLen),
                    (const char *Flags, size_t FlagsLen),
                    (const char *Producer, size_t ProducerLen),
                    (const char *SplitName, size_t SplitNameLen),
                    (const char *SysRoot, size_t SysRootLen),
                    (const char *SDK, size_t SDKLen),
                    (const char *Str, unsigned Length),
                    (const char *UniqueIdentifier, size_t UniqueIdentifierLen),
                    (const char *Str, unsigned SLen),
                    (const char *Str, size_t SLen),
                    (const char *Name, unsigned SLen),
                    (const char *Name, size_t SLen),
                    (const char *Linkage, size_t LinkLen)
{
    $result = C.CString($1)
}

%typemap(goout)     (const char *Name, size_t NameLen),
                    (const char *Filename, size_t FilenameLen),
                    (const char *Directory, size_t DirectoryLen),
                    (const char *UniqueId, size_t UniqueIdLen),
                    (const char *LinkageName, size_t LinkageNameLen),
                    (const char *Flags, size_t FlagsLen),
                    (const char *Producer, size_t ProducerLen),
                    (const char *SplitName, size_t SplitNameLen),
                    (const char *SysRoot, size_t SysRootLen),
                    (const char *SDK, size_t SDKLen),
                    (const char *Str, unsigned Length),
                    (const char *UniqueIdentifier, size_t UniqueIdentifierLen),
                    (const char *Str, unsigned SLen),
                    (const char *Str, size_t SLen),
                    (const char *Name, unsigned SLen),
                    (const char *Name, size_t SLen),
                    (const char *Linkage, size_t LinkLen)
{
    $result = C.GoString($1)
}


%typemap(in)        char *,
                    const char *
{
    $1 = $input;
}

%typemap(out)       char *,
                    const char *
{
    $result = $1;
}

%typemap(freearg)   char *,
                    const char *
{
    free($input);
}

%typemap(imtype)    char *,
                    const char * "*C.char"

%typemap(gotype)    char *,
                    const char * "string"

%typemap(goin)      char *,
                    const char *
{
    $result = C.CString($input)
}

%typemap(goout)     char *,
                    const char *
{
    $result = C.GoString($input)
}

// Handle C array <--> Go slice
%typemap(gotype) (LLVMTypeRef *ParamTypes, unsigned ParamCount),
                 (LLVMTypeRef *ElementTypes, unsigned ElementCount),
                 (LLVMTypeRef *TypeParams, unsigned TypeParamCount)
                 "[]LLVMTypeRef";

%typemap(imtype) (LLVMTypeRef *ParamTypes, unsigned ParamCount),
                 (LLVMTypeRef *ElementTypes, unsigned ElementCount),
                 (LLVMTypeRef *TypeParams, unsigned TypeParamCount)
                 "[]C.LLVMTypeRef";

%typemap(gotype) (LLVMValueRef *ConstantVals, unsigned Count),
                 (LLVMValueRef *ConstantVals, unsigned Length),
                 (LLVMValueRef *ConstantVals, uint64_t Length),
                 (LLVMValueRef *ScalarConstantVals, unsigned Size),
                 (LLVMValueRef *ConstantIndices, unsigned NumIndices),
                 (LLVMValueRef *Vals, unsigned Count),
                 (LLVMValueRef *IncomingValues, unsigned Count),
                 (LLVMValueRef *RetVals, unsigned N),
                 (LLVMValueRef *Args, unsigned NumArgs),
                 (LLVMValueRef *Indices, unsigned NumIndices),
                 (LLVMValueRef *ScalarConstantVals, unsigned Size)
                 "[]LLVMValueRef";

%typemap(imtype) (LLVMValueRef *ConstantVals, unsigned Count),
                 (LLVMValueRef *ConstantVals, unsigned Length),
                 (LLVMValueRef *ConstantVals, uint64_t Length),
                 (LLVMValueRef *ScalarConstantVals, unsigned Size),
                 (LLVMValueRef *ConstantIndices, unsigned NumIndices),
                 (LLVMValueRef *Vals, unsigned Count),
                 (LLVMValueRef *IncomingValues, unsigned Count),
                 (LLVMValueRef *RetVals, unsigned N),
                 (LLVMValueRef *Args, unsigned NumArgs),
                 (LLVMValueRef *Indices, unsigned NumIndices),
                 (LLVMValueRef *ScalarConstantVals, unsigned Size)
                 "[]C.LLVMValueRef";

%typemap(gotype) (LLVMMetadataRef *ParameterTypes, unsigned NumParameterTypes),
                 (LLVMMetadataRef *Elements, unsigned NumElements),
                 (LLVMMetadataRef *Data, unsigned NumElements),
                 (LLVMMetadataRef *Subscripts, unsigned NumSubscripts),
                 (LLVMMetadataRef *Data, size_t NumElements),
                 (LLVMMetadataRef *MDs, size_t Count)
                 "[]LLVMMetadataRef";

%typemap(imtype) (LLVMMetadataRef *ParameterTypes, unsigned NumParameterTypes),
                 (LLVMMetadataRef *Elements, unsigned NumElements),
                 (LLVMMetadataRef *Data, unsigned NumElements),
                 (LLVMMetadataRef *Subscripts, unsigned NumSubscripts),
                 (LLVMMetadataRef *Data, size_t NumElements),
                 (LLVMMetadataRef *MDs, size_t Count)
                 "[]C.LLVMMetadataRef";

%typemap(goin)  (LLVMTypeRef *ParamTypes, unsigned ParamCount),
                (LLVMTypeRef *ElementTypes, unsigned ElementCount),
                (LLVMTypeRef *TypeParams, unsigned TypeParamCount),
                (LLVMValueRef *ConstantVals, unsigned Count),
                (LLVMValueRef *ConstantVals, unsigned Length),
                (LLVMValueRef *ConstantVals, uint64_t Length),
                (LLVMValueRef *ScalarConstantVals, unsigned Size),
                (LLVMValueRef *ConstantIndices, unsigned NumIndices),
                (LLVMValueRef *Vals, unsigned Count),
                (LLVMValueRef *IncomingValues, unsigned Count),
                (LLVMValueRef *RetVals, unsigned N),
                (LLVMValueRef *Args, unsigned NumArgs),
                (LLVMValueRef *Indices, unsigned NumIndices),
                (LLVMValueRef *ScalarConstantVals, unsigned Size),
                (LLVMMetadataRef *ParameterTypes, unsigned NumParameterTypes),
                (LLVMMetadataRef *Elements, unsigned NumElements),
                (LLVMMetadataRef *Data, unsigned NumElements),
                (LLVMMetadataRef *Subscripts, unsigned NumSubscripts),
                (LLVMMetadataRef *Data, size_t NumElements),
                (LLVMMetadataRef *MDs, size_t Count)
{
   $result = make([]C.$*1_type, 0, len($input))
   for _, val := range $input {
        if val != nil {
            $result = append($result, (C.$*1_type)(unsafe.Pointer(val)))
       } else {
            $result = append($result, (C.$*1_type)(unsafe.Pointer(nil)))
       }
   }
}

%typemap(in)    (LLVMTypeRef *ParamTypes, unsigned ParamCount),
                (LLVMTypeRef *ElementTypes, unsigned ElementCount),
                (LLVMTypeRef *TypeParams, unsigned TypeParamCount),
                (LLVMValueRef *ConstantVals, unsigned Count),
                (LLVMValueRef *ConstantVals, unsigned Length),
                (LLVMValueRef *ConstantVals, uint64_t Length),
                (LLVMValueRef *ScalarConstantVals, unsigned Size),
                (LLVMValueRef *ConstantIndices, unsigned NumIndices),
                (LLVMValueRef *Vals, unsigned Count),
                (LLVMValueRef *IncomingValues, unsigned Count),
                (LLVMValueRef *RetVals, unsigned N),
                (LLVMValueRef *Args, unsigned NumArgs),
                (LLVMValueRef *Indices, unsigned NumIndices),
                (LLVMValueRef *ScalarConstantVals, unsigned Size),
                (LLVMMetadataRef *ParameterTypes, unsigned NumParameterTypes),
                (LLVMMetadataRef *Elements, unsigned NumElements),
                (LLVMMetadataRef *Data, unsigned NumElements),
                (LLVMMetadataRef *Subscripts, unsigned NumSubscripts),
                (LLVMMetadataRef *Data, size_t NumElements),
                (LLVMMetadataRef *MDs, size_t Count)
{
    $1 = ($*1_type*)$input.array;
    $2 = ($2_type)$input.len;
}

%define LLVM_TYPEMAP(OPAQUE, TYPE)
%typemap(gotype) TYPE "TYPE";
%typemap(gotype) TYPE * "*TYPE";
%typemap(imtype) TYPE "C.TYPE"
%typemap(goin) TYPE "$result = C.$type(unsafe.Pointer($input))"
%typemap(goout) TYPE "$result = TYPE(unsafe.Pointer($1))"
%typemap(in) TYPE "$1 = $input;"
%typemap(out) TYPE "$result = $1;"
%insert(go_wrapper) %{
type TYPE C.TYPE
%}
%enddef

LLVM_TYPEMAP(LLVMOpaqueContext, LLVMContextRef)
LLVM_TYPEMAP(LLVMOpaqueModule, LLVMModuleRef)
LLVM_TYPEMAP(LLVMOpaqueType, LLVMTypeRef)
LLVM_TYPEMAP(LLVMOpaqueValue, LLVMValueRef)
LLVM_TYPEMAP(LLVMOpaqueBasicBlock, LLVMBasicBlockRef)
LLVM_TYPEMAP(LLVMOpaqueMetadata, LLVMMetadataRef)
LLVM_TYPEMAP(LLVMOpaqueNamedMDNode, LLVMNamedMDNodeRef)
LLVM_TYPEMAP(LLVMOpaqueValueMetadataEntry, LLVMValueMetadataEntry)
LLVM_TYPEMAP(LLVMOpaqueBuilder, LLVMBuilderRef)
LLVM_TYPEMAP(LLVMOpaqueDIBuilder, LLVMDIBuilderRef)
LLVM_TYPEMAP(LLVMOpaqueModuleProvide, LLVMModuleProviderRef)
LLVM_TYPEMAP(LLVMOpaquePassManager, LLVMPassManagerRef)
LLVM_TYPEMAP(LLVMOpaquePassRegistry, LLVMPassRegistryRef)
LLVM_TYPEMAP(LLVMOpaqueUse, LLVMUseRef)
LLVM_TYPEMAP(LLVMOpaqueAttributeRef, LLVMAttributeRef)
LLVM_TYPEMAP(LLVMOpaqueDiagnosticInfo, LLVMDiagnosticInfoRef)
LLVM_TYPEMAP(LLVMComdat, LLVMComdatRef)
LLVM_TYPEMAP(LLVMOpaqueModuleFlagEntry, LLVMModuleFlagEntry)
LLVM_TYPEMAP(LLVMOpaqueJITEventListener, LLVMJITEventListenerRef)
LLVM_TYPEMAP(LLVMOpaqueBinary, LLVMBinaryRef)
LLVM_TYPEMAP(LLVMOpaqueTargetMachine, LLVMTargetMachineRef)
LLVM_TYPEMAP(LLVMOpaqueTarget, LLVMTargetRef)
LLVM_TYPEMAP(LLVMOpaqueTargetData, LLVMTargetDataRef)
LLVM_TYPEMAP(LLVMOpaqueTargetLibraryInfotData, LLVMTargetLibraryInfoRef)
LLVM_TYPEMAP(LLVMOpaquePassBuilderOptions, LLVMPassBuilderOptionsRef)
LLVM_TYPEMAP(LLVMOpaquePassManagerBuilder, LLVMPassManagerBuilderRef)
LLVM_TYPEMAP(LLVMOpaqueSectionIterator, LLVMSectionIteratorRef)
LLVM_TYPEMAP(LLVMOpaqueSymbolIterator, LLVMSymbolIteratorRef)
LLVM_TYPEMAP(LLVMOpaqueRelocationIterator, LLVMRelocationIteratorRef)
LLVM_TYPEMAP(LLVMOpaqueMemoryBuffer, LLVMMemoryBufferRef)
LLVM_TYPEMAP(LLVMOpaqueObjectFile, LLVMObjectFileRef)
LLVM_TYPEMAP(LLVMOpaqueError, LLVMErrorRef)
//%insert(go_wrapper) %{
//func (l LLVMErrorRef) Error() string {
//    return GetErrorMessage(l)
//}
//%}

// Handle some APIs manually
%ignore LLVMGetTargetFromTriple;
%ignore LLVMAddIncoming;
%ignore LLVMGetParamTypes;
%ignore LLVMGetStructElementTypes;
%ignore LLVMGetValueName2;

// Strip redundant "LLVM" from function calls
%rename("%(strip:[LLVM])s") "";

%include "llvm-c/ExternC.h"
%include "llvm-c/Types.h"
%include "llvm-c/Error.h"
%include "llvm-c/ErrorHandling.h"
%include "llvm-c/Deprecated.h"
%include "llvm-c/Analysis.h"
%include "llvm-c/BitReader.h"
%include "llvm-c/BitWriter.h"
%include "llvm-c/blake3.h"
%include "llvm-c/Comdat.h"
%include "llvm-c/Core.h"
%include "llvm-c/DebugInfo.h"
%include "llvm-c/Disassembler.h"
%include "llvm-c/DisassemblerTypes.h"
%include "llvm-c/Initialization.h"
%include "llvm-c/IRReader.h"
%include "llvm-c/Linker.h"
%include "llvm-c/Object.h"
%include "llvm-c/Support.h"
%include "llvm-c/Target.h"
%include "llvm-c/TargetMachine.h"
%include "llvm-c/Transforms/InstCombine.h"
%include "llvm-c/Transforms/IPO.h"
%include "llvm-c/Transforms/PassBuilder.h"
%include "llvm-c/Transforms/PassManagerBuilder.h"
%include "llvm-c/Transforms/Scalar.h"
%include "llvm-c/Transforms/Utils.h"
%include "llvm-c/Transforms/Vectorize.h"