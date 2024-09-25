%module mlir

#define _WIN32

// Remove Mlir prefix from functions and types
%rename("%(regex:/^(Mlir|mlir)(.*)/\\2/)s") "";

%header %{
#include <stdbool.h>
//#include "mlir-c/Dialect/Async.h"
//#include "mlir-c/Dialect/ControlFlow.h"
#include "mlir-c/Dialect/Func.h"
//#include "mlir-c/Dialect/GPU.h"
//#include "mlir-c/Dialect/Linalg.h"
#include "mlir-c/Dialect/LLVM.h"
//#include "mlir-c/Dialect/MLProgram.h"
//#include "mlir-c/Dialect/PDL.h"
//#include "mlir-c/Dialect/Quant.h"
//#include "mlir-c/Dialect/SCF.h"
//#include "mlir-c/Dialect/Shape.h"
//#include "mlir-c/Dialect/SparseTensor.h"
//#include "mlir-c/Dialect/Tensor.h" 
//#include "mlir-c/Dialect/Transform.h"

#include "mlir-c/Support.h"
//#include "mlir-c/AffineExpr.h"
//#include "mlir-c/AffineMap.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
//#include "mlir-c/Conversion.h"
//#include "mlir-c/Debug.h"
//#include "mlir-c/Diagnostics.h"
//#include "mlir-c/ExecutionEngine.h"
//#include "mlir-c/IntegerSet.h"
//#include "mlir-c/Interfaces.h"
#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
//#include "mlir-c/RegisterEverything.h"
//#include "mlir-c/Transforms.h"

#include "Go-c/llvm/Passes.h"
#include "Go-c/mlir/Dialects.h"
#include "Go-c/mlir/Enums.h"
#include "Go-c/mlir/Operations.h"
#include "Go-c/mlir/Types.h"
%}

%insert(cgo_comment_typedefs) %{
#include <stdlib.h>
#include <string.h>
//#include "mlir-c/Dialect/Async.h"
//#include "mlir-c/Dialect/ControlFlow.h"
#include "mlir-c/Dialect/Func.h"
//#include "mlir-c/Dialect/GPU.h"
//#include "mlir-c/Dialect/Linalg.h"
#include "mlir-c/Dialect/LLVM.h"
//#include "mlir-c/Dialect/MLProgram.h"
//#include "mlir-c/Dialect/PDL.h"
//#include "mlir-c/Dialect/Quant.h"
//#include "mlir-c/Dialect/SCF.h"
//#include "mlir-c/Dialect/Shape.h"
//#include "mlir-c/Dialect/SparseTensor.h"
//#include "mlir-c/Dialect/Tensor.h"
//#include "mlir-c/Dialect/Transform.h"

//#include "mlir-c/AffineExpr.h"
//#include "mlir-c/AffineMap.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
//#include "mlir-c/Conversion.h"
//#include "mlir-c/Debug.h"
//#include "mlir-c/Diagnostics.h"
//#include "mlir-c/ExecutionEngine.h"
//#include "mlir-c/IntegerSet.h"
//#include "mlir-c/Interfaces.h"
#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
//#include "mlir-c/RegisterEverything.h"
#include "mlir-c/Support.h"
//#include "mlir-c/Transforms.h"

#include "Go-c/llvm/Passes.h"
#include "Go-c/mlir/Dialects.h"
#include "Go-c/mlir/Enums.h"
#include "Go-c/mlir/Operations.h"
#include "Go-c/mlir/Types.h"
%}

%include inttypes.i
%include typemaps.i

%ignore mlirStringRefCreate;
%ignore mlirStringRefCreateFromCString;
%ignore mlirStringRefEqual;

%define MLIR_STRING_SLICE_TYPEMAP(N_PARAM, ARR_PARAM)
%typemap(gotype) (intptr_t N_PARAM, MlirStringRef *ARR_PARAM) "[]string";
%typemap(imtype) (intptr_t N_PARAM, MlirStringRef *ARR_PARAM) "[]C.MlirStringRef";
%typemap(goin) (intptr_t N_PARAM, MlirStringRef *ARR_PARAM) %{
    $result = make([]C.MlirStringRef, 0, len($input))
    for _, val := range $input {
        strVal := C.mlirStringRefCreateFromCString(C.CString(val))
        $result = append($result, strVal)
    }
%}

%typemap(in) (intptr_t N_PARAM, MlirStringRef *ARR_PARAM) %{
    $1 = ($1_type)$input.len;
    $2 = (MlirStringRef*)$input.array;
%}
%enddef

MLIR_STRING_SLICE_TYPEMAP(nMethodNames, methodNames)

%typemap(gotype) MlirStringRef "string"
%typemap(imtype) MlirStringRef "C.MlirStringRef"
%typemap(goin) MlirStringRef %{
    $result = C.mlirStringRefCreateFromCString(C.CString($input))
%}
%typemap(goout) MlirStringRef %{
    $result = C.GoStringN($input.data, C.int($input.length))
%}
%typemap(in) MlirStringRef %{
    $1 = $input;
%}
%typemap(out) MlirStringRef %{
    $result = $1;
%}

%define MLIR_SLICE_TYPEMAP(TYPE, GOTYPE, N_PARAM, ARR_PARAM)
%typemap(gotype) (intptr_t N_PARAM, TYPE *ARR_PARAM) "[]GOTYPE";
%typemap(imtype) (intptr_t N_PARAM, TYPE *ARR_PARAM) "[]C.TYPE";
%typemap(goin) (intptr_t N_PARAM, TYPE *ARR_PARAM) %{
   $result = make([]C.TYPE, 0, len($input))
   for _, val := range $input {
        if val != nil {
            $result = append($result, *((*C.TYPE)(unsafe.Pointer(getSwigcptr(val)))))
       } else {
            $result = append($result, *((*C.TYPE)(unsafe.Pointer(uintptr(0)))))
       }
   }
%}

%typemap(in) (intptr_t N_PARAM, TYPE *ARR_PARAM) %{
    $1 = ($1_type)$input.len;
    $2 = (TYPE*)$input.array;
%}

%typemap(gotype) (intptr_t N_PARAM, TYPE const *ARR_PARAM) "[]GOTYPE";
%typemap(imtype) (intptr_t N_PARAM, TYPE const *ARR_PARAM) "[]C.TYPE";
%typemap(goin) (intptr_t N_PARAM, TYPE const *ARR_PARAM) %{
    $result = make([]C.TYPE, 0, len($input))
    for _, val := range $input {
        if val != nil {
            $result = append($result, *((*C.TYPE)(unsafe.Pointer(getSwigcptr(val)))))
        } else {
            $result = append($result, *((*C.TYPE)(unsafe.Pointer(uintptr(0)))))
        }
    }
%}

%typemap(in) (intptr_t N_PARAM, TYPE const *ARR_PARAM) %{
    $1 = ($1_type)$input.len;
    $2 = (TYPE*)$input.array;
%}
%enddef

MLIR_SLICE_TYPEMAP(MlirAffineExpr, AffineExpr, nAffineExprs, affineExprs)

MLIR_SLICE_TYPEMAP(MlirAttribute, Attribute, numElements, elements)
MLIR_SLICE_TYPEMAP(MlirAttribute, Attribute, nElements, elements)
MLIR_SLICE_TYPEMAP(MlirAttribute, Attribute, nEntries, entries)
MLIR_SLICE_TYPEMAP(MlirAttribute, Attribute, nTypes, types)
MLIR_SLICE_TYPEMAP(MlirAttribute, Attribute, nNames, names)
MLIR_SLICE_TYPEMAP(MlirAttribute, Attribute, nTags, tags)
MLIR_SLICE_TYPEMAP(MlirAttribute, Attribute, numReferences, references)

MLIR_SLICE_TYPEMAP(MlirBlock, Block, n, successors)

MLIR_SLICE_TYPEMAP(MlirNamedAttribute, NamedAttribute, n, attributes)
MLIR_SLICE_TYPEMAP(MlirNamedAttribute, NamedAttribute, nAttributes, attributes)
MLIR_SLICE_TYPEMAP(MlirNamedAttribute, NamedAttribute, numElements, elements)

MLIR_SLICE_TYPEMAP(MlirLocation, Location, nLocations, locations)

MLIR_SLICE_TYPEMAP(MlirOperation, Operation, nOperands, operands)

MLIR_SLICE_TYPEMAP(MlirRegion, Region, n, regions)

MLIR_SLICE_TYPEMAP(MlirType, Type, n, results)
MLIR_SLICE_TYPEMAP(MlirType, Type, numInputs, inputs)
MLIR_SLICE_TYPEMAP(MlirType, Type, nInputs, inputs)
MLIR_SLICE_TYPEMAP(MlirType, Type, numResults, results)
MLIR_SLICE_TYPEMAP(MlirType, Type, nResults, results)
MLIR_SLICE_TYPEMAP(MlirType, Type, nArgumentTypes, argumentTypes)
MLIR_SLICE_TYPEMAP(MlirType, Type, nFieldTypes, fieldTypes)
MLIR_SLICE_TYPEMAP(MlirType, Type, nArgs, args)
MLIR_SLICE_TYPEMAP(MlirType, Type, nResultTypes, resultTypes)
MLIR_SLICE_TYPEMAP(MlirType, Type, nMemberTypes, memberTypes)
MLIR_SLICE_TYPEMAP(MlirType, Type, nMethods, methods)
MLIR_SLICE_TYPEMAP(MlirType, Type, nFields, fields)

MLIR_SLICE_TYPEMAP(MlirValue, Value, n, operands)
MLIR_SLICE_TYPEMAP(MlirValue, Value, nOperands, operands)
MLIR_SLICE_TYPEMAP(MlirValue, Value, nDestOperands, destOperands)
MLIR_SLICE_TYPEMAP(MlirValue, Value, nTrueDestOperands, trueDestOperands)
MLIR_SLICE_TYPEMAP(MlirValue, Value, nFalseDestOperands, falseDestOperands)
MLIR_SLICE_TYPEMAP(MlirValue, Value, nIndices, indices)
MLIR_SLICE_TYPEMAP(MlirValue, Value, nDynamicIndices, dynamicIndices)
MLIR_SLICE_TYPEMAP(MlirValue, Value, nValues, values)
MLIR_SLICE_TYPEMAP(MlirValue, Value, nArgs, args)

%define MLIR_PRIMITIVE_SLICE_TYPEMAP(TYPE, GOTYPE, N_PARAM, ARR_PARAM)
%typemap(gotype) (intptr_t N_PARAM, TYPE *ARR_PARAM) "[]GOTYPE";
%typemap(gotype) (intptr_t N_PARAM, TYPE *ARR_PARAM) "[]GOTYPE";
%typemap(imtype) (intptr_t N_PARAM, TYPE *ARR_PARAM) "[]C.TYPE";
%typemap(goin) (intptr_t N_PARAM, TYPE *ARR_PARAM) %{
    $result = make([]C.TYPE, 0, len($input))
    for _, val := range $input {
        $result = append($result, C.TYPE(val))
    }
%}

%typemap(in) (intptr_t N_PARAM, TYPE *ARR_PARAM) %{
    $1 = ($1_type)$input.len;
    $2 = (TYPE*)$input.array;
%}

%typemap(gotype) (intptr_t N_PARAM, TYPE const *ARR_PARAM) "[]GOTYPE";
%typemap(imtype) (intptr_t N_PARAM, TYPE const *ARR_PARAM) "[]C.TYPE";
%typemap(in) (intptr_t N_PARAM, TYPE const *ARR_PARAM) "[]C.TYPE";
%typemap(goin) (intptr_t N_PARAM, TYPE const *ARR_PARAM) %{
   $result = make([]C.TYPE, 0, len($input))
   for _, val := range $input {
       $result = append($result, C.TYPE(val))
   }
%}

%typemap(in) (intptr_t N_PARAM, TYPE const *ARR_PARAM) %{
    $1 = ($1_type)$input.len;
    $2 = (TYPE*)$input.array;
%}
%enddef

MLIR_PRIMITIVE_SLICE_TYPEMAP(bool, bool, size, values)
MLIR_PRIMITIVE_SLICE_TYPEMAP(int8_t, int8, size, values)
MLIR_PRIMITIVE_SLICE_TYPEMAP(int16_t, int16, size, values)
MLIR_PRIMITIVE_SLICE_TYPEMAP(int32_t, int32, size, values)
MLIR_PRIMITIVE_SLICE_TYPEMAP(int64_t, int64, size, values)
MLIR_PRIMITIVE_SLICE_TYPEMAP(float, float32, size, values)
MLIR_PRIMITIVE_SLICE_TYPEMAP(double, float64, size, values)
MLIR_PRIMITIVE_SLICE_TYPEMAP(int32_t, int32, nConstIndices, constIndices)

// Handle LLVM typs

%go_import("omibyte.io/sigo/llvm")

%define LLVM_TYPEMAP(TYPE)

%typemap(gotype) TYPE "llvm.TYPE"
%typemap(imtype) TYPE "C.TYPE"
%typemap(goin) TYPE "$result = C.$type(unsafe.Pointer($input.Swigcptr()))"
%typemap(goout) TYPE "$result.SetSwigcptr(uintptr(unsafe.Pointer($1)))"
%typemap(in) TYPE "$1 = $input;"
%typemap(out) TYPE "$result = $1;"

%enddef

LLVM_TYPEMAP(LLVMContextRef)
LLVM_TYPEMAP(LLVMModuleRef)
LLVM_TYPEMAP(LLVMTypeRef)
LLVM_TYPEMAP(LLVMValueRef)
LLVM_TYPEMAP(LLVMBasicBlockRef)
LLVM_TYPEMAP(LLVMMetadataRef)
LLVM_TYPEMAP(LLVMNamedMDNodeRef)
LLVM_TYPEMAP(LLVMValueMetadataEntry)
LLVM_TYPEMAP(LLVMBuilderRef)
LLVM_TYPEMAP(LLVMDIBuilderRef)
LLVM_TYPEMAP(LLVMModuleProviderRef)
LLVM_TYPEMAP(LLVMPassManagerRef)
LLVM_TYPEMAP(LLVMUseRef)
LLVM_TYPEMAP(LLVMAttributeRef)
LLVM_TYPEMAP(LLVMDiagnosticInfoRef)
LLVM_TYPEMAP(LLVMComdatRef)
LLVM_TYPEMAP(LLVMModuleFlagEntry)
LLVM_TYPEMAP(LLVMJITEventListenerRef)
LLVM_TYPEMAP(LLVMBinaryRef)
LLVM_TYPEMAP(LLVMTargetMachineRef)
LLVM_TYPEMAP(LLVMTargetRef)
LLVM_TYPEMAP(LLVMTargetDataRef)
LLVM_TYPEMAP(LLVMTargetLibraryInfoRef)
LLVM_TYPEMAP(LLVMPassBuilderOptionsRef)
LLVM_TYPEMAP(LLVMSectionIteratorRef)
LLVM_TYPEMAP(LLVMSymbolIteratorRef)
LLVM_TYPEMAP(LLVMRelocationIteratorRef)
LLVM_TYPEMAP(LLVMMemoryBufferRef)
LLVM_TYPEMAP(LLVMObjectFileRef)
LLVM_TYPEMAP(LLVMErrorRef)

%include "mlir-c/Support.h"

//%include "mlir-c/AffineExpr.h"
//%include "mlir-c/AffineMap.h"
%include "mlir-c/BuiltinAttributes.h"
%include "mlir-c/BuiltinTypes.h"
//%include "mlir-c/Conversion.h"
//%include "mlir-c/Debug.h"
//%include "mlir-c/Diagnostics.h"
//%include "mlir-c/ExecutionEngine.h"
//%include "mlir-c/IntegerSet.h"
//%include "mlir-c/Interfaces.h"
%include "mlir-c/IR.h"
%include "mlir-c/Pass.h"
//%include "mlir-c/RegisterEverything.h"
//%include "mlir-c/Transforms.h"

//%include "mlir-c/Dialect/Async.h"
//%include "mlir-c/Dialect/ControlFlow.h"
%include "mlir-c/Dialect/Func.h"
//%include "mlir-c/Dialect/GPU.h"
//%include "mlir-c/Dialect/Linalg.h"
%include "mlir-c/Dialect/LLVM.h"
//%include "mlir-c/Dialect/MLProgram.h"
//%include "mlir-c/Dialect/PDL.h"
//%include "mlir-c/Dialect/Quant.h"
//%include "mlir-c/Dialect/SCF.h"
//%include "mlir-c/Dialect/Shape.h"
//%include "mlir-c/Dialect/SparseTensor.h"
//%include "mlir-c/Dialect/Tensor.h"
//%include "mlir-c/Dialect/Transform.h"

%include "Go-c/llvm/Passes.h"
%include "Go-c/mlir/Dialects.h"
%include "Go-c/mlir/Enums.h"
%include "Go-c/mlir/Operations.h"
%include "Go-c/mlir/Types.h"