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
%}

%insert(cgo_comment_typedefs) %{
#include <stdlib.h>
#include "llvm-c/Types.h"
#include "llvm-c/Target.h"
#include "llvm-c/TargetMachine.h"
%}

%typemap(gotype) LLVMTypeKind "LLVMTypeKind"
%typemap(gotype) LLVMIntPredicate "LLVMIntPredicate"
%typemap(gotype) LLVMRealPredicate "LLVMRealPredicate"

%ignore LLVMTypeKind;
%ignore LLVMIntPredicate;
%ignore LLVMRealPredicate;

/*
// Typemaps for Go bool
%typemap(in) bool {
  $input = $1 ? 1 : 0;
}
%typemap(out) bool {
  $result = $1 != 0;
}
*/

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
            $result = append($result, (C.$*1_type)(unsafe.Pointer(val.Swigcptr())))
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

// Handle passing normal ref types
%typemap(imtype)
              LLVMMemoryBufferRef,
              LLVMContextRef,
              LLVMModuleRef,
              LLVMTypeRef,
              LLVMValueRef,
              LLVMBasicBlockRef,
              LLVMMetadataRef,
              LLVMNamedMDNodeRef,
              LLVMBuilderRef,
              LLVMDIBuilderRef,
              LLVMModuleProviderRef,
              LLVMPassManagerRef,
              LLVMPassRegistryRef,
              LLVMUseRef,
              LLVMAttributeRef,
              LLVMDiagnosticInfoRef,
              LLVMComdatRef,
              LLVMJITEventListenerRef,
              LLVMBinaryRef,
              LLVMTargetMachineRef,
              LLVMTargetRef,
              LLVMTargetDataRef,
              LLVMTargetLibraryInfoRef "C.$type"

%typemap(goin)
            LLVMMemoryBufferRef,
            LLVMContextRef,
            LLVMModuleRef,
            LLVMTypeRef,
            LLVMValueRef,
            LLVMBasicBlockRef,
            LLVMMetadataRef,
            LLVMNamedMDNodeRef,
            LLVMBuilderRef,
            LLVMDIBuilderRef,
            LLVMModuleProviderRef,
            LLVMPassManagerRef,
            LLVMPassRegistryRef,
            LLVMUseRef,
            LLVMAttributeRef,
            LLVMDiagnosticInfoRef,
            LLVMComdatRef,
            LLVMJITEventListenerRef,
            LLVMBinaryRef,
            LLVMTargetMachineRef,
            LLVMTargetRef,
            LLVMTargetDataRef,
            LLVMTargetLibraryInfoRef
{
    if $input == nil {
        $result = C.$type(unsafe.Pointer(uintptr(0)))
    } else {
        $result = C.$type(unsafe.Pointer($input.Swigcptr()))
    }
}

// Handle return from C wrapper to Go
%typemap(goout, match="out")
            LLVMMemoryBufferRef,
            LLVMContextRef,
            LLVMModuleRef,
            LLVMTypeRef,
            LLVMValueRef,
            LLVMBasicBlockRef,
            LLVMMetadataRef,
            LLVMNamedMDNodeRef,
            LLVMBuilderRef,
            LLVMDIBuilderRef,
            LLVMModuleProviderRef,
            LLVMPassManagerRef,
            LLVMPassRegistryRef,
            LLVMUseRef,
            LLVMAttributeRef,
            LLVMDiagnosticInfoRef,
            LLVMComdatRef,
            LLVMJITEventListenerRef,
            LLVMBinaryRef,
            LLVMTargetMachineRef,
            LLVMTargetRef,
            LLVMTargetDataRef,
            LLVMTargetLibraryInfoRef
{
    if $1 == nil {
        $result = nil
    } else {
        $result = Swigcptr$1_type(unsafe.Pointer($1))
    }
}

%typemap(out)
              LLVMMemoryBufferRef,
              LLVMContextRef,
              LLVMModuleRef,
              LLVMTypeRef,
              LLVMValueRef,
              LLVMBasicBlockRef,
              LLVMMetadataRef,
              LLVMNamedMDNodeRef,
              LLVMBuilderRef,
              LLVMDIBuilderRef,
              LLVMModuleProviderRef,
              LLVMPassManagerRef,
              LLVMPassRegistryRef,
              LLVMUseRef,
              LLVMAttributeRef,
              LLVMDiagnosticInfoRef,
              LLVMComdatRef,
              LLVMJITEventListenerRef,
              LLVMBinaryRef,
              LLVMTargetMachineRef,
              LLVMTargetRef,
              LLVMTargetDataRef,
              LLVMTargetLibraryInfoRef
{
    $result = $1;
}

%typemap(in)
              LLVMMemoryBufferRef,
              LLVMContextRef,
              LLVMModuleRef,
              LLVMTypeRef,
              LLVMValueRef,
              LLVMBasicBlockRef,
              LLVMMetadataRef,
              LLVMNamedMDNodeRef,
              LLVMBuilderRef,
              LLVMDIBuilderRef,
              LLVMModuleProviderRef,
              LLVMPassManagerRef,
              LLVMPassRegistryRef,
              LLVMUseRef,
              LLVMAttributeRef,
              LLVMDiagnosticInfoRef,
              LLVMComdatRef,
              LLVMJITEventListenerRef,
              LLVMBinaryRef,
              LLVMTargetMachineRef,
              LLVMTargetRef,
              LLVMTargetDataRef,
              LLVMTargetLibraryInfoRef
{
    $1 = $input;
}

// Handle some APIs manually

%ignore LLVMGetTargetFromTriple;
%ignore LLVMAddIncoming;
%ignore LLVMGetParamTypes;
%ignore LLVMGetStructElementTypes;
%ignore LLVMGetValueName2;

// Strip redundant "LLVM" from function calls
%{
typedef int                                     LLVMBool;
typedef unsigned                                LLVMDWARFTypeEncoding;
typedef struct LLVMOpaqueMemoryBuffer           *LLVMMemoryBufferRef;
typedef struct LLVMOpaqueContext                *LLVMContextRef;
typedef struct LLVMOpaqueModule                 *LLVMModuleRef;
typedef struct LLVMOpaqueType                   *LLVMTypeRef;
typedef struct LLVMOpaqueValue                  *LLVMValueRef;
typedef struct LLVMOpaqueBasicBlock             *LLVMBasicBlockRef;
typedef struct LLVMOpaqueMetadata               *LLVMMetadataRef;
typedef struct LLVMOpaqueNamedMDNode            *LLVMNamedMDNodeRef;
typedef struct LLVMOpaqueValueMetadataEntry     LLVMValueMetadataEntry;
typedef struct LLVMOpaqueBuilder                *LLVMBuilderRef;
typedef struct LLVMOpaqueDIBuilder              *LLVMDIBuilderRef;
typedef struct LLVMOpaqueModuleProvider         *LLVMModuleProviderRef;
typedef struct LLVMOpaquePassManager            *LLVMPassManagerRef;
typedef struct LLVMOpaquePassRegistry           *LLVMPassRegistryRef;
typedef struct LLVMOpaqueUse                    *LLVMUseRef;
typedef struct LLVMOpaqueAttributeRef           *LLVMAttributeRef;
typedef struct LLVMOpaqueDiagnosticInfo         *LLVMDiagnosticInfoRef;
typedef struct LLVMComdat                       *LLVMComdatRef;
typedef struct LLVMOpaqueModuleFlagEntry        LLVMModuleFlagEntry;
typedef struct LLVMOpaqueJITEventListener       *LLVMJITEventListenerRef;
typedef struct LLVMOpaqueBinary                 *LLVMBinaryRef;

typedef struct LLVMOpaqueTargetMachine          *LLVMTargetMachineRef;
typedef struct LLVMTarget                       *LLVMTargetRef;
typedef struct LLVMOpaqueTargetData             *LLVMTargetDataRef;
typedef struct LLVMOpaqueTargetLibraryInfotData *LLVMTargetLibraryInfoRef;
%}

%rename("%(strip:[LLVM])s") "";

%include "llvm-c/ExternC.h"
//%include "llvm-c/Types.h" // Excluded to correct the resulting type names
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
%include "llvm-c/Error.h"
//%include "llvm-c/ExecutionEngine.h"
%include "llvm-c/Initialization.h"
%include "llvm-c/IRReader.h"
%include "llvm-c/Linker.h"
//%include "llvm-c/LLJIT.h"
//%include "llvm-c/lto.h"
%include "llvm-c/Object.h"
//%include "llvm-c/Orc.h"
//%include "llvm-c/OrcEE.h"
//%include "llvm-c/Remarks.h"
%include "llvm-c/Support.h"
%include "llvm-c/Target.h"
%include "llvm-c/TargetMachine.h"