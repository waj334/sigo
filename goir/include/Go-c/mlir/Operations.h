#ifndef GO_C_OPERATIONS_H
#define GO_C_OPERATIONS_H

#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/IR.h>
#include <mlir-c/Support.h>

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateAddCOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                        MlirLocation location);

MlirOperation mlirGoCreateAddFOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                        MlirLocation location);

MlirOperation mlirGoCreateAddIOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                        MlirLocation location);

MlirOperation mlirGoCreateAddStrOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                          MlirLocation location);

MlirOperation mlirGoCreateAndOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                       MlirLocation location);

MlirOperation mlirGoCreateAndNotOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                          MlirLocation location);

MlirOperation mlirGoCreateCmpCOperation(MlirContext context, MlirType resultType, MlirAttribute predicate, MlirValue x,
                                        MlirValue y, MlirLocation location);

MlirOperation mlirGoCreateCmpFOperation(MlirContext context, MlirType resultType, MlirAttribute predicate, MlirValue x,
                                        MlirValue y, MlirLocation location);

MlirOperation mlirGoCreateCmpIOperation(MlirContext context, MlirType resultType, MlirAttribute predicate, MlirValue x,
                                        MlirValue y, MlirLocation location);

MlirOperation mlirGoCreateDivCOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                        MlirLocation location);

MlirOperation mlirGoCreateDivFOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                        MlirLocation location);

MlirOperation mlirGoCreateDivSIOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                         MlirLocation location);

MlirOperation mlirGoCreateDivUIOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                         MlirLocation location);

MlirOperation mlirGoCreateMulCOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                        MlirLocation location);

MlirOperation mlirGoCreateMulFOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                        MlirLocation location);

MlirOperation mlirGoCreateMulIOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                        MlirLocation location);

MlirOperation mlirGoCreateOrOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                      MlirLocation location);

MlirOperation mlirGoCreateRemFOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                        MlirLocation location);

MlirOperation mlirGoCreateRemSIOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                         MlirLocation location);

MlirOperation mlirGoCreateRemUIOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                         MlirLocation location);

MlirOperation mlirGoCreateShlOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                       MlirLocation location);

MlirOperation mlirGoCreateShrUIOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                         MlirLocation location);

MlirOperation mlirGoCreateShrSIOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                         MlirLocation location);

MlirOperation mlirGoCreateSubCOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                        MlirLocation location);

MlirOperation mlirGoCreateSubFOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                        MlirLocation location);

MlirOperation mlirGoCreateSubIOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                        MlirLocation location);

MlirOperation mlirGoCreateXorOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                       MlirLocation location);

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateComplementOperation(MlirContext context, MlirValue x, MlirLocation location);

MlirOperation mlirGoCreateNegCOperation(MlirContext context, MlirValue x, MlirLocation location);

MlirOperation mlirGoCreateNegFOperation(MlirContext context, MlirValue x, MlirLocation location);

MlirOperation mlirGoCreateNegIOperation(MlirContext context, MlirValue x, MlirLocation location);

MlirOperation mlirGoCreateNotOperation(MlirContext context, MlirValue x, MlirLocation location);

MlirOperation mlirGoCreateRecvOperation(MlirContext context, MlirValue x, bool commaOk, MlirType resultType,
                                        MlirLocation location);

//===----------------------------------------------------------------------===//
// Map Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateMapUpdateOperation(MlirContext context, MlirValue map, MlirValue key, MlirValue value,
                                             MlirLocation location);

MlirOperation mlirGoCreateMapLookupOperation(MlirContext context, MlirType resultType, MlirValue map, MlirValue key,
                                             bool hasOk,
                                             MlirLocation location);

//===----------------------------------------------------------------------===//
// Memory Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateAllocaOperation(MlirContext context, MlirType resultType, MlirType elementType,
                                          MlirValue *numElements, bool isHeap, MlirLocation location);

void mlirGoAllocaOperationSetName(MlirOperation op, MlirStringRef name);

void mlirGoAllocaOperationSetIsHeap(MlirOperation op, bool isHeap);

MlirOperation mlirGoCreateLoadOperation(MlirContext context, MlirValue x, MlirType resultType, MlirLocation location);

MlirOperation mlirGoCreateVolatileLoadOperation(MlirContext context, MlirValue x, MlirType resultType,
                                                MlirLocation location);

MlirOperation mlirGoCreateAtomicLoadOperation(MlirContext context, MlirValue x, MlirType resultType,
                                              MlirLocation location);

MlirOperation mlirGoCreateStoreOperation(MlirContext context, MlirValue value, MlirValue address,
                                         MlirLocation location);

MlirOperation mlirGoCreateVolatileStoreOperation(MlirContext context, MlirValue value, MlirValue address,
                                                 MlirLocation location);

MlirOperation mlirGoCreateAtomicStoreOperation(MlirContext context, MlirValue value, MlirValue address,
                                               MlirLocation location);

MlirOperation mlirGoCreateGepOperation(MlirContext context, MlirValue addr, MlirType baseType, intptr_t nConstIndices,
                                       int32_t *constIndices, intptr_t nDynamicIndices, MlirValue *dynamicIndices,
                                       MlirType type, MlirLocation location);

MlirOperation mlirGoCreateGlobalOperation(MlirContext context, MlirAttribute *linkage, MlirStringRef symbol,
                                          MlirType type, MlirLocation location);

MlirOperation mlirGoCreateYieldOperation(MlirContext context, MlirValue value, MlirLocation location);

MlirOperation mlirGoCreateSliceOperation(MlirContext context, MlirValue input, MlirValue *low, MlirValue *high,
                                         MlirValue *max, MlirType resultType, MlirLocation location);

MlirOperation mlirGoCreateAddressOfOperation(MlirContext context, MlirStringRef symbol, MlirType resultType,
                                             MlirLocation location);

//===----------------------------------------------------------------------===//
// Slice Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateSliceAddrOperation(MlirContext context, MlirType resultType, MlirValue slice, MlirValue index,
                                             MlirLocation location);


//===----------------------------------------------------------------------===//
// Struct Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateExtractOperation(MlirContext context, uint64_t index, MlirType fieldType,
                                           MlirValue structValue, MlirLocation location);

MlirOperation mlirGoCreateInsertOperation(MlirContext context, uint64_t index, MlirValue value, MlirValue structValue,
                                          MlirType structType, MlirLocation location);

//===----------------------------------------------------------------------===//
// Constant Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateConstantOperation(MlirContext context, MlirAttribute value, MlirType type,
                                            MlirLocation location);

//===----------------------------------------------------------------------===//
// Casting Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateBitcastOperation(MlirContext context, MlirValue value, MlirType type, MlirLocation location);

MlirOperation mlirGoCreateComplexExtendOperation(MlirContext context, MlirValue value, MlirType type,
                                                 MlirLocation location);

MlirOperation mlirGoCreateComplexTruncateOperation(MlirContext context, MlirValue value, MlirType type,
                                                   MlirLocation location);

MlirOperation mlirGoCreateIntToPtrOperation(MlirContext context, MlirValue value, MlirType type, MlirLocation location);

MlirOperation mlirGoCreatePtrToIntOperation(MlirContext context, MlirValue value, MlirType type, MlirLocation location);

MlirOperation mlirGoCreateFloatTruncateOperation(MlirContext context, MlirValue value, MlirType type,
                                                 MlirLocation location);

MlirOperation mlirGoCreateIntTruncateOperation(MlirContext context, MlirValue value, MlirType type,
                                               MlirLocation location);

MlirOperation mlirGoCreateFloatExtendOperation(MlirContext context, MlirValue value, MlirType type,
                                               MlirLocation location);

MlirOperation mlirGoCreateSignedExtendOperation(MlirContext context, MlirValue value, MlirType type,
                                                MlirLocation location);

MlirOperation mlirGoCreateZeroExtendOperation(MlirContext context, MlirValue value, MlirType type,
                                              MlirLocation location);

MlirOperation mlirGoCreateFloatToUnsignedIntOperation(MlirContext context, MlirValue value, MlirType type,
                                                      MlirLocation location);

MlirOperation mlirGoCreateFloatToSignedIntOperation(MlirContext context, MlirValue value, MlirType type,
                                                    MlirLocation location);

MlirOperation mlirGoCreateUnsignedIntToFloatOperation(MlirContext context, MlirValue value, MlirType type,
                                                      MlirLocation location);

MlirOperation mlirGoCreateSignedIntToFloatOperation(MlirContext context, MlirValue value, MlirType type,
                                                    MlirLocation location);

MlirOperation mlirGoCreateFunctionToPointerOperation(MlirContext context, MlirValue value, MlirType type,
                                                     MlirLocation location);

MlirOperation mlirGoCreatePointerToFunctionOperation(MlirContext context, MlirValue value, MlirType type,
                                                     MlirLocation location);

MlirOperation mlirGoCreateChangeInterfaceOperation(MlirContext context, MlirValue value, MlirType type,
                                                   MlirLocation location);

MlirOperation mlirGoCreateTypeAssertOperation(MlirContext context, MlirValue value, MlirType type,
                                              MlirLocation location);

//===----------------------------------------------------------------------===//
// Function Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoGetFunction(MlirContext context, MlirStringRef symbol, MlirType type, MlirLocation location);

//===----------------------------------------------------------------------===//
// Intrinsic Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateDeclareTypeOperation(MlirContext context, MlirType type, MlirAttribute attributes,
                                               MlirLocation location);

MlirOperation mlirGoCreateTypeInfoOperation(MlirContext context, MlirType resultType, MlirType type,
                                            MlirLocation location);

//===----------------------------------------------------------------------===//
// Builtin Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreatePanicOperation(MlirContext context, MlirValue value, MlirBlock *recoverBlock,
                                         MlirLocation location);

MlirOperation mlirGoCreateRecoverOperation(MlirContext context, MlirType type, MlirLocation location);

//===----------------------------------------------------------------------===//
// Atomic Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateAtomicAddIOperation(MlirContext context, MlirType resultType, MlirValue addr, MlirValue delta,
                                              MlirLocation location);

MlirOperation mlirGoCreateAtomicCompareAndSwapOperation(MlirContext context, MlirType resultType, MlirValue addr,
                                                        MlirValue old, MlirValue value, MlirLocation location);

MlirOperation mlirGoCreateAtomicSwapOperation(MlirContext context, MlirType resultType, MlirValue addr, MlirValue value,
                                              MlirLocation location);

//===----------------------------------------------------------------------===//
// Control Flow Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateBranchOperation(MlirContext context, MlirBlock dest, intptr_t nDestOperands,
                                          MlirValue *destOperands, MlirLocation location);

MlirOperation mlirGoCreateCondBranchOperation(MlirContext context, MlirValue condition, MlirBlock trueDest,
                                              intptr_t nTrueDestOperands, MlirValue *trueDestOperands,
                                              MlirBlock falseDest, intptr_t nFalseDestOperands,
                                              MlirValue *falseDestOperands, MlirLocation location);

MlirOperation mlirGoCreateReturnOperation(MlirContext context, intptr_t nOperands, MlirValue *operands,
                                          MlirLocation location);

//===----------------------------------------------------------------------===//
// Call Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateCallOperation(MlirContext context, MlirStringRef callee, intptr_t nResultTypes,
                                        MlirType *resultTypes, intptr_t nOperands, MlirValue *operands,
                                        MlirLocation location);

MlirOperation mlirGoCreateCallIndirectOperation(MlirContext context, MlirValue callee, intptr_t nResultTypes,
                                                MlirType *resultTypes, intptr_t nOperands, MlirValue *operands,
                                                MlirLocation location);

MlirOperation mlirGoCreateDeferOperation(MlirContext context, MlirValue fn, MlirAttribute *method, intptr_t nArgs,
                                         MlirValue *args, MlirLocation location);

MlirOperation mlirGoCreateGoOperation(MlirContext context, MlirValue fn, MlirType signature, MlirStringRef method, intptr_t nArgs,
                                      MlirValue *args, MlirLocation location);

MlirOperation mlirGoCreateInterfaceCall(MlirContext context, MlirStringRef callee, MlirType signature, MlirValue value,
                                        intptr_t nArgs, MlirValue *args, MlirLocation location);

MlirOperation mlirGoCreateRuntimeCallOperation(MlirContext context, MlirStringRef callee, intptr_t nResultTypes,
                                               MlirType *resultTypes, intptr_t nOperands, MlirValue *operands,
                                               MlirLocation location);

//===----------------------------------------------------------------------===//
// Value Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateZeroOperation(MlirContext context, MlirType type, MlirLocation location);

MlirOperation mlirGoCreateComplexOperation(MlirContext context, MlirType type, MlirValue real, MlirValue imag,
                                           MlirLocation location);

MlirOperation mlirGoCreateImagOperation(MlirContext context, MlirType type, MlirValue value, MlirLocation location);

MlirOperation mlirGoCreateRealOperation(MlirContext context, MlirType type, MlirValue value, MlirLocation location);

MlirOperation mlirGoCreateMakeChanOperation(MlirContext context, MlirType resultType, MlirValue *capacity,
                                            MlirLocation location);

MlirOperation mlirGoCreateMakeMapOperation(MlirContext context, MlirType resultType, MlirValue *capacity,
                                           MlirLocation location);

MlirOperation mlirGoCreateMakeSliceOperation(MlirContext context, MlirType resultType, MlirValue length,
                                             MlirValue *capacity, MlirLocation location);

MlirOperation mlirGoCreateMakeInterfaceOperation(MlirContext context, MlirType resultType, MlirType type,
                                                 MlirValue value,
                                                 MlirLocation location);


#ifdef __cplusplus
}
#endif

#endif // GO_C_OPERATIONS_H
