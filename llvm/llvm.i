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
#include "llvm-c/ExecutionEngine.h"
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
%ignore LLVMTypeKind;

// Typemaps for Go bool
%typemap(in) bool {
  $input = $1 ? 1 : 0;
}
%typemap(out) bool {
  $result = $1 != 0;
}

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
                    (const char *LinkageName, size_t LinkageNameLen)
{
    $1 = $input;
    $2 = strlen($input);
}

%typemap(out)       (const char *Name, size_t NameLen),
                    (const char *Filename, size_t FilenameLen),
                    (const char *Directory, size_t DirectoryLen),
                    (const char *UniqueId, size_t UniqueIdLen),
                    (const char *LinkageName, size_t LinkageNameLen)
{
    $result = $1;
}

%typemap(freearg)   (const char *Name, size_t NameLen),
                    (const char *Filename, size_t FilenameLen),
                    (const char *Directory, size_t DirectoryLen),
                    (const char *UniqueId, size_t UniqueIdLen),
                    (const char *LinkageName, size_t LinkageNameLen)
{
    free($input);
}

%typemap(imtype)    (const char *Name, size_t NameLen),
                    (const char *Filename, size_t FilenameLen),
                    (const char *Directory, size_t DirectoryLen),
                    (const char *UniqueId, size_t UniqueIdLen),
                    (const char *LinkageName, size_t LinkageNameLen)"*C.char"

%typemap(gotype)    (const char *Name, size_t NameLen),
                    (const char *Filename, size_t FilenameLen),
                    (const char *Directory, size_t DirectoryLen),
                    (const char *UniqueId, size_t UniqueIdLen),
                    (const char *LinkageName, size_t LinkageNameLen) "string"

%typemap(goin)      (const char *Name, size_t NameLen),
                    (const char *Filename, size_t FilenameLen),
                    (const char *Directory, size_t DirectoryLen),
                    (const char *UniqueId, size_t UniqueIdLen),
                    (const char *LinkageName, size_t LinkageNameLen)
{
    $result = C.CString($1)

    // NOTE: This crashes for some reason
    //defer C.free(unsafe.Pointer($result))
}

%typemap(goout)     (const char *Name, size_t NameLen),
                    (const char *Filename, size_t FilenameLen),
                    (const char *Directory, size_t DirectoryLen),
                    (const char *UniqueId, size_t UniqueIdLen),
                    (const char *LinkageName, size_t LinkageNameLen)
{
    $result = C.GoString($1)

    // NOTE: This crashes for some reason
    //defer C.free(unsafe.Pointer($1))
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

    // NOTE: This crashes for some reason
    //defer C.free(unsafe.Pointer($result))
}

%typemap(goout)     char *,
                    const char *
{
    $result = C.GoString($input)

    // NOTE: This crashes for some reason
    //defer C.free(unsafe.Pointer($input))
}

// *********** LLVMTypeRef
%typemap(gotype) (LLVMTypeRef *ParamTypes, unsigned ParamCount),
                 (LLVMTypeRef *ElementTypes, unsigned ElementCount),
                 (LLVMTypeRef *TypeParams, unsigned TypeParamCount) "[]LLVMTypeRef";

%typemap(imtype) (LLVMTypeRef *ParamTypes, unsigned ParamCount),
                 (LLVMTypeRef *ElementTypes, unsigned ElementCount),
                 (LLVMTypeRef *TypeParams, unsigned TypeParamCount) "*C.LLVMTypeRef"

%typemap(goin) (LLVMTypeRef *ParamTypes, unsigned ParamCount),
               (LLVMTypeRef *ElementTypes, unsigned ElementCount),
               (LLVMTypeRef *TypeParams, unsigned TypeParamCount),
               (LLVMTypeRef *TypeParams, unsigned TypeParamCount){
   $result = (*C.LLVMTypeRef)(C.malloc(C.size_t(len($input)) * C.size_t(unsafe.Sizeof(uintptr(0)))))
   defer C.free(unsafe.Pointer($result))
   if $result == nil {
       panic("Failed to allocate memory for C array.")
   }

   for i, val := range $input {
       *(*C.LLVMTypeRef)(unsafe.Pointer(uintptr(unsafe.Pointer($result)) + uintptr(i)*unsafe.Sizeof(uintptr(0)))) = (C.LLVMTypeRef)(unsafe.Pointer(val.Swigcptr()))
   }
}

%typemap(in) (LLVMTypeRef *ParamTypes, unsigned ParamCount),
             (LLVMTypeRef *ElementTypes, unsigned ElementCount),
             (LLVMTypeRef *TypeParams, unsigned TypeParamCount){}

%typemap(arginit) (LLVMTypeRef *ParamTypes, unsigned ParamCount),
                  (LLVMTypeRef *ElementTypes, unsigned ElementCount),
                  (LLVMTypeRef *TypeParams, unsigned TypeParamCount) %{
  $1 = NULL;
  $2 = 0;
%}

%typemap(freearg) (LLVMTypeRef *ParamTypes, unsigned ParamCount),
                  (LLVMTypeRef *ElementTypes, unsigned ElementCount),
                  (LLVMTypeRef *TypeParams, unsigned TypeParamCount){}

// *********** LLVMValueRef
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
                 (LLVMValueRef *ScalarConstantVals, unsigned Size) "[]LLVMValueRef";

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
                 (LLVMValueRef *ScalarConstantVals, unsigned Size) "*C.LLVMValueRef"

%typemap(goin) (LLVMValueRef *ConstantVals, unsigned Count),
               (LLVMValueRef *ConstantVals, unsigned Length),
               (LLVMValueRef *ConstantVals, uint64_t Length),
               (LLVMValueRef *ScalarConstantVals, unsigned Size),
               (LLVMValueRef *ConstantIndices, unsigned NumIndices),
               (LLVMValueRef *Vals, unsigned Count),
               (LLVMValueRef *IncomingValues, unsigned Count),
               (LLVMValueRef *RetVals, unsigned N),
               (LLVMValueRef *Args, unsigned NumArgs),
               (LLVMValueRef *Indices, unsigned NumIndices),
               (LLVMValueRef *ScalarConstantVals, unsigned Size){
   $result = (*C.LLVMValueRef)(C.malloc(C.size_t(len($input)) * C.size_t(unsafe.Sizeof(uintptr(0)))))
   defer C.free(unsafe.Pointer($result))
   if $result == nil {
       panic("Failed to allocate memory for C array.")
   }

   for i, val := range $input {
       *(*C.LLVMValueRef)(unsafe.Pointer(uintptr(unsafe.Pointer($result)) + uintptr(i)*unsafe.Sizeof(uintptr(0)))) = (C.LLVMValueRef)(unsafe.Pointer(val.Swigcptr()))
   }
}

%typemap(in) (LLVMValueRef *ConstantVals, unsigned Count),
             (LLVMValueRef *ConstantVals, unsigned Length),
             (LLVMValueRef *ConstantVals, uint64_t Length),
             (LLVMValueRef *ScalarConstantVals, unsigned Size),
             (LLVMValueRef *ConstantIndices, unsigned NumIndices),
             (LLVMValueRef *Vals, unsigned Count),
             (LLVMValueRef *IncomingValues, unsigned Count),
             (LLVMValueRef *RetVals, unsigned N),
             (LLVMValueRef *Args, unsigned NumArgs),
             (LLVMValueRef *Indices, unsigned NumIndices),
             (LLVMValueRef *ScalarConstantVals, unsigned Size){}

%typemap(arginit) (LLVMValueRef *ConstantVals, unsigned Count),
                  (LLVMValueRef *ConstantVals, unsigned Length),
                  (LLVMValueRef *ConstantVals, uint64_t Length),
                  (LLVMValueRef *ScalarConstantVals, unsigned Size),
                  (LLVMValueRef *ConstantIndices, unsigned NumIndices),
                  (LLVMValueRef *Vals, unsigned Count),
                  (LLVMValueRef *IncomingValues, unsigned Count),
                  (LLVMValueRef *RetVals, unsigned N),
                  (LLVMValueRef *Args, unsigned NumArgs),
                  (LLVMValueRef *Indices, unsigned NumIndices),
                  (LLVMValueRef *ScalarConstantVals, unsigned Size) %{
  $1 = NULL;
  $2 = 0;
%}

%typemap(freearg) (LLVMValueRef *ConstantVals, unsigned Count),
                  (LLVMValueRef *ConstantVals, unsigned Length),
                  (LLVMValueRef *ConstantVals, uint64_t Length),
                  (LLVMValueRef *ScalarConstantVals, unsigned Size),
                  (LLVMValueRef *ConstantIndices, unsigned NumIndices),
                  (LLVMValueRef *Vals, unsigned Count),
                  (LLVMValueRef *IncomingValues, unsigned Count),
                  (LLVMValueRef *RetVals, unsigned N),
                  (LLVMValueRef *Args, unsigned NumArgs),
                  (LLVMValueRef *Indices, unsigned NumIndices),
                  (LLVMValueRef *ScalarConstantVals, unsigned Size){}

// *********** LLVMMetadataRef
%typemap(gotype) (LLVMMetadataRef *ParameterTypes, unsigned NumParameterTypes),
                 (LLVMMetadataRef *Elements, unsigned NumElements),
                 (LLVMMetadataRef *Data, unsigned NumElements),
                 (LLVMMetadataRef *Subscripts, unsigned NumSubscripts) "[]LLVMMetadataRef";

%typemap(imtype) (LLVMMetadataRef *ParameterTypes, unsigned NumParameterTypes),
                 (LLVMMetadataRef *Elements, unsigned NumElements),
                 (LLVMMetadataRef *Data, unsigned NumElements),
                 (LLVMMetadataRef *Subscripts, unsigned NumSubscripts) "*C.LLVMMetadataRef"

%typemap(goin) (LLVMMetadataRef *ParameterTypes, unsigned NumParameterTypes),
               (LLVMMetadataRef *Elements, unsigned NumElements),
               (LLVMMetadataRef *Data, unsigned NumElements),
               (LLVMMetadataRef *Subscripts, unsigned NumSubscripts) {
   $result = (*C.LLVMMetadataRef)(C.malloc(C.size_t(len($input)) * C.size_t(unsafe.Sizeof(uintptr(0)))))
   defer C.free(unsafe.Pointer($result))
   if $result == nil {
       panic("Failed to allocate memory for C array.")
   }

   for i, val := range $input {
       *(*C.LLVMMetadataRef)(unsafe.Pointer(uintptr(unsafe.Pointer($result)) + uintptr(i)*unsafe.Sizeof(uintptr(0)))) = (C.LLVMMetadataRef)(unsafe.Pointer(val.Swigcptr()))
   }
}

%typemap(in) (LLVMMetadataRef *ParameterTypes, unsigned NumParameterTypes),
             (LLVMMetadataRef *Elements, unsigned NumElements),
             (LLVMMetadataRef *Data, unsigned NumElements),
             (LLVMMetadataRef *Subscripts, unsigned NumSubscripts) {}

%typemap(arginit) (LLVMMetadataRef *ParameterTypes, unsigned NumParameterTypes),
                  (LLVMMetadataRef *Elements, unsigned NumElements),
                  (LLVMMetadataRef *Data, unsigned NumElements),
                  (LLVMMetadataRef *Subscripts, unsigned NumSubscripts) %{
  $1 = NULL;
  $2 = 0;
%}

%typemap(freearg) (LLVMMetadataRef *ParameterTypes, unsigned NumParameterTypes),
                  (LLVMMetadataRef *Elements, unsigned NumElements),
                  (LLVMMetadataRef *Data, unsigned NumElements),
                  (LLVMMetadataRef *Subscripts, unsigned NumSubscripts) {}


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
              LLVMTargetLibraryInfoRef "unsafe.Pointer"

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
        $result = unsafe.Pointer(uintptr(0))
    } else {
        $result = unsafe.Pointer($input.Swigcptr())
    }
}

%typemap(goout, match="out") LLVMMemoryBufferRef {
    if $1 == nil {
        $result = nil
    } else {
        $result = SwigcptrLLVMMemoryBufferRef($1)
    }
}

%typemap(goout, match="out") LLVMContextRef {
    if $1 == nil {
        $result = nil
    } else {
        $result = SwigcptrLLVMContextRef($1)
    }
}

%typemap(goout, match="out") LLVMModuleRef {
    if $1 == nil {
        $result = nil
    } else {
        $result = SwigcptrLLVMModuleRef($1)
    }
}

%typemap(goout, match="out") LLVMTypeRef {
    if $1 == nil {
        $result = nil
    } else {
        $result = SwigcptrLLVMTypeRef($1)
    }
}

%typemap(goout, match="out") LLVMValueRef {
    if $1 == nil {
        $result = nil
    } else {
        $result = SwigcptrLLVMValueRef($1)
    }
}

%typemap(goout, match="out") LLVMBasicBlockRef {
    if $1 == nil {
        $result = nil
    } else {
        $result = SwigcptrLLVMBasicBlockRef($1)
    }
}

%typemap(goout, match="out") LLVMMetadataRef {
    if $1 == nil {
        $result = nil
    } else {
        $result = SwigcptrLLVMMetadataRef($1)
    }
}

%typemap(goout, match="out") LLVMNamedMDNodeRef {
    if $1 == nil {
        $result = nil
    } else {
        $result = SwigcptrLLVMNamedMDNodeRef($1)
    }
}

%typemap(goout, match="out") LLVMBuilderRef {
    if $1 == nil {
        $result = nil
    } else {
        $result = SwigcptrLLVMBuilderRef($1)
    }
}

%typemap(goout, match="out") LLVMDIBuilderRef {
    if $1 == nil {
        $result = nil
    } else {
        $result = SwigcptrLLVMDIBuilderRef($1)
    }
}

%typemap(goout, match="out") LLVMModuleProviderRef {
    if $1 == nil {
        $result = nil
    } else {
        $result = SwigcptrLLVMModuleProviderRef($1)
    }
}

%typemap(goout, match="out") LLVMPassManagerRef {
    if $1 == nil {
        $result = nil
    } else {
        $result = SwigcptrLLVMPassManagerRef($1)
    }
}

%typemap(goout, match="out") LLVMPassRegistryRef {
    if $1 == nil {
        $result = nil
    } else {
        $result = SwigcptrLLVMPassRegistryRef($1)
    }
}

%typemap(goout, match="out") LLVMUseRef {
    if $1 == nil {
        $result = nil
    } else {
        $result = SwigcptrLLVMUseRef($1)
    }
}

%typemap(goout, match="out") LLVMAttributeRef {
    if $1 == nil {
        $result = nil
    } else {
        $result = SwigcptrLLVMAttributeRef($1)
    }
}

%typemap(goout, match="out") LLVMDiagnosticInfoRef {
    if $1 == nil {
        $result = nil
    } else {
        $result = SwigcptrLLVMDiagnosticInfoRef($1)
    }
}

%typemap(goout, match="out") LLVMComdatRef {
    if $1 == nil {
        $result = nil
    } else {
        $result = SwigcptrLLVMComdatRef($1)
    }
}

%typemap(goout, match="out") LLVMJITEventListenerRef {
    if $1 == nil {
        $result = nil
    } else {
        $result = SwigcptrLLVMJITEventListenerRef($1)
    }
}

%typemap(goout, match="out") LLVMBinaryRef {
    if $1 == nil {
        $result = nil
    } else {
        $result = SwigcptrLLVMBinaryRef($1)
    }
}

%typemap(goout, match="out") LLVMTargetMachineRef {
    if $1 == nil {
        $result = nil
    } else {
        $result = SwigcptrLLVMTargetMachineRef($1)
    }
}

%typemap(goout, match="out") LLVMTargetRef {
    if $1 == nil {
        $result = nil
    } else {
        $result = SwigcptrLLVMTargetRef($1)
    }
}

%typemap(goout, match="out") LLVMTargetDataRef {
    if $1 == nil {
        $result = nil
    } else {
        $result = SwigcptrLLVMTargetDataRef($1)
    }
}

%typemap(goout, match="out") LLVMTargetLibraryInfoRef {
    if $1 == nil {
        $result = nil
    } else {
        $result = SwigcptrLLVMTargetLibraryInfoRef($1)
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

// Handle LLVMGetTargetFromTriple manually

%ignore LLVMGetTargetFromTriple;

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

%include argnames.i

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
%include "llvm-c/ExecutionEngine.h"
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