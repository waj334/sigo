#ifndef GO_C_OPERATIONS_H
#define GO_C_OPERATIONS_H

#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/IR.h>
#include <mlir-c/Support.h>

#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

  //===----------------------------------------------------------------------===//
  // ASM Operations
  //===----------------------------------------------------------------------===//

  MlirOperation mlirGoCreateInlineAssemblyOperation(
    const MlirContext context,
    const MlirAttribute asmStr,
    const int nConstraints,
    const MlirAttribute* constraints,
    const int nRegisterClobbers,
    const MlirAttribute* registerClobbers,
    const int nOperands,
    const MlirValue* operands,
    const MlirLocation location);

  //===----------------------------------------------------------------------===//
  // Binary Operations
  //===----------------------------------------------------------------------===//

  MlirOperation mlirGoCreateAddCOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirValue x,
    const MlirValue y,
    const MlirLocation location);

  MlirOperation mlirGoCreateAddFOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirValue x,
    const MlirValue y,
    const MlirLocation location);

  MlirOperation mlirGoCreateAddIOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirValue x,
    const MlirValue y,
    const MlirLocation location);

  MlirOperation mlirGoCreateAddStrOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirValue x,
    const MlirValue y,
    const MlirLocation location);

  MlirOperation mlirGoCreateAndOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirValue x,
    const MlirValue y,
    const MlirLocation location);

  MlirOperation mlirGoCreateAndNotOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirValue x,
    const MlirValue y,
    const MlirLocation location);

  MlirOperation mlirGoCreateCmpCOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirAttribute predicate,
    const MlirValue x,
    const MlirValue y,
    const MlirLocation location);

  MlirOperation mlirGoCreateCmpFOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirAttribute predicate,
    const MlirValue x,
    const MlirValue y,
    const MlirLocation location);

  MlirOperation mlirGoCreateCmpIOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirAttribute predicate,
    const MlirValue x,
    const MlirValue y,
    const MlirLocation location);

  MlirOperation mlirGoCreateCmpInterfaceOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirValue x,
    const MlirValue y,
    const MlirLocation location);

  MlirOperation mlirGoCreateCmpStringOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirAttribute predicate,
    const MlirValue x,
    const MlirValue y,
    const MlirLocation location);

  MlirOperation mlirGoCreateCmpNilOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirValue x,
    const MlirLocation location);

  MlirOperation mlirGoCreateDivCOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirValue x,
    const MlirValue y,
    const MlirLocation location);

  MlirOperation mlirGoCreateDivFOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirValue x,
    const MlirValue y,
    const MlirLocation location);

  MlirOperation mlirGoCreateDivSIOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirValue x,
    const MlirValue y,
    const MlirLocation location);

  MlirOperation mlirGoCreateDivUIOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirValue x,
    const MlirValue y,
    const MlirLocation location);

  MlirOperation mlirGoCreateMulCOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirValue x,
    const MlirValue y,
    const MlirLocation location);

  MlirOperation mlirGoCreateMulFOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirValue x,
    const MlirValue y,
    const MlirLocation location);

  MlirOperation mlirGoCreateMulIOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirValue x,
    const MlirValue y,
    const MlirLocation location);

  MlirOperation mlirGoCreateOrOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirValue x,
    const MlirValue y,
    const MlirLocation location);

  MlirOperation mlirGoCreateRemFOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirValue x,
    const MlirValue y,
    const MlirLocation location);

  MlirOperation mlirGoCreateRemSIOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirValue x,
    const MlirValue y,
    const MlirLocation location);

  MlirOperation mlirGoCreateRemUIOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirValue x,
    const MlirValue y,
    const MlirLocation location);

  MlirOperation mlirGoCreateShlOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirValue x,
    const MlirValue y,
    const MlirLocation location);

  MlirOperation mlirGoCreateShrUIOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirValue x,
    const MlirValue y,
    const MlirLocation location);

  MlirOperation mlirGoCreateShrSIOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirValue x,
    const MlirValue y,
    const MlirLocation location);

  MlirOperation mlirGoCreateSubCOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirValue x,
    const MlirValue y,
    const MlirLocation location);

  MlirOperation mlirGoCreateSubFOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirValue x,
    const MlirValue y,
    const MlirLocation location);

  MlirOperation mlirGoCreateSubIOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirValue x,
    const MlirValue y,
    const MlirLocation location);

  MlirOperation mlirGoCreateXorOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirValue x,
    const MlirValue y,
    const MlirLocation location);

  //===----------------------------------------------------------------------===//
  // Unary Operations
  //===----------------------------------------------------------------------===//

  MlirOperation
  mlirGoCreateComplementOperation(const MlirContext context, const MlirValue x, const MlirLocation location);

  MlirOperation mlirGoCreateNegCOperation(const MlirContext context, const MlirValue x, const MlirLocation location);

  MlirOperation mlirGoCreateNegFOperation(const MlirContext context, const MlirValue x, const MlirLocation location);

  MlirOperation mlirGoCreateNegIOperation(const MlirContext context, const MlirValue x, const MlirLocation location);

  MlirOperation mlirGoCreateNotOperation(const MlirContext context, const MlirValue x, const MlirLocation location);

  //===----------------------------------------------------------------------===//
  // Map Operations
  //===----------------------------------------------------------------------===//

  MlirOperation mlirGoCreateMapAddrOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirValue map,
    const MlirValue key,
    const MlirLocation location);

  MlirOperation mlirGoCreateMapLookupOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirValue map,
    const MlirValue key,
    const bool hasOk,
    const MlirLocation location);

  MlirOperation mlirGoCreateMapRangeOp(
    const MlirContext context,
    const MlirValue value,
    const MlirBlock bodyDest,
    const MlirBlock exitDest,
    const MlirLocation location);

  //===----------------------------------------------------------------------===//
  // Memory Operations
  //===----------------------------------------------------------------------===//

  MlirOperation mlirGoCreateAllocaOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirType elementType,
    const int numElements,
    const bool isHeap,
    const MlirLocation location);

  void mlirGoAllocaOperationSetName(const MlirOperation op, const MlirStringRef name);

  void mlirGoAllocaOperationSetIsHeap(const MlirOperation op, const bool isHeap);

  MlirOperation mlirGoCreateLoadOperation(
    const MlirContext context,
    const MlirValue x,
    const MlirType resultType,
    const MlirLocation location);

  MlirOperation mlirGoCreateVolatileLoadOperation(
    const MlirContext context,
    const MlirValue x,
    const MlirType resultType,
    const MlirLocation location);

  MlirOperation mlirGoCreateAtomicLoadOperation(
    const MlirContext context,
    const MlirValue x,
    const MlirType resultType,
    const MlirLocation location);

  MlirOperation mlirGoCreateStoreOperation(
    const MlirContext context,
    const MlirValue value,
    const MlirValue address,
    const MlirLocation location);

  MlirOperation mlirGoCreateVolatileStoreOperation(
    const MlirContext context,
    const MlirValue value,
    const MlirValue address,
    const MlirLocation location);

  MlirOperation mlirGoCreateAtomicStoreOperation(
    const MlirContext context,
    const MlirValue value,
    const MlirValue address,
    const MlirLocation location);

  MlirOperation mlirGoCreateGepOperation(
    const MlirContext context,
    const MlirValue addr,
    const MlirType baseType,
    const int nConstIndices,
    const int32_t* constIndices,
    const int nDynamicIndices,
    const MlirValue* dynamicIndices,
    const int nIndexFlags,
    const bool* indexFlags,
    const MlirType type,
    const MlirLocation location);

  MlirOperation mlirGoCreateGlobalOperation(
    const MlirContext context,
    const MlirAttribute linkage,
    const MlirStringRef symbol,
    const MlirStringRef section,
    const MlirType type,
    const MlirLocation location);

  MlirOperation
  mlirGoCreateYieldOperation(const MlirContext context, const MlirValue value, const MlirLocation location);

  MlirOperation mlirGoCreateSliceOperation(
    const MlirContext context,
    const MlirValue input,
    const MlirValue low,
    const MlirValue high,
    const MlirValue max,
    const MlirType resultType,
    const MlirLocation location);

  MlirOperation mlirGoCreateAddressOfOperation(
    const MlirContext context,
    const MlirStringRef symbol,
    const MlirType resultType,
    const MlirLocation location);

  MlirOperation
  mlirGoCreateNilPointerCheckOperation(const MlirContext context, const MlirValue addr, const MlirLocation location);

  //===----------------------------------------------------------------------===//
  // Slice Operations
  //===----------------------------------------------------------------------===//

  MlirOperation mlirGoCreateSliceAddrOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirValue slice,
    const MlirValue index,
    const MlirLocation location);

  //===----------------------------------------------------------------------===//
  // String Operations
  //===----------------------------------------------------------------------===//

  MlirOperation mlirGoCreateStringAddrOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirValue slice,
    const MlirValue index,
    const MlirLocation location);

  MlirOperation mlirGoCreateStringRangeOp(
    const MlirContext context,
    const MlirValue value,
    const MlirBlock bodyDest,
    const MlirBlock exitDest,
    const MlirLocation location);

  //===----------------------------------------------------------------------===//
  // Struct Operations
  //===----------------------------------------------------------------------===//

  MlirOperation mlirGoCreateExtractOperation(
    const MlirContext context,
    const uint64_t index,
    const MlirType fieldType,
    const MlirValue structValue,
    const MlirLocation location);

  MlirOperation mlirGoCreateInsertOperation(
    const MlirContext context,
    const uint64_t index,
    const MlirValue value,
    const MlirValue structValue,
    const MlirType structType,
    const MlirLocation location);

  //===----------------------------------------------------------------------===//
  // Constant Operations
  //===----------------------------------------------------------------------===//

  MlirOperation mlirGoCreateConstantOperation(
    const MlirContext context,
    const MlirAttribute value,
    const MlirAttribute symbol,
    const MlirType type,
    const MlirLocation location);

  MlirOperation mlirGoCreateGlobalConstantOperation(
    const MlirContext context,
    const MlirAttribute value,
    const MlirAttribute symbol,
    const MlirLocation location);

  void mlirGoGlobalConstantOperationAddBody(const MlirOperation op, const MlirBlock body);

  //===----------------------------------------------------------------------===//
  // Casting Operations
  //===----------------------------------------------------------------------===//

  MlirOperation mlirGoCreateBitcastOperation(
    const MlirContext context,
    const MlirValue value,
    const MlirType type,
    const MlirLocation location);

  MlirOperation mlirGoCreateComplexExtendOperation(
    const MlirContext context,
    const MlirValue value,
    const MlirType type,
    const MlirLocation location);

  MlirOperation mlirGoCreateComplexTruncateOperation(
    const MlirContext context,
    const MlirValue value,
    const MlirType type,
    const MlirLocation location);

  MlirOperation mlirGoCreateIntToPtrOperation(
    const MlirContext context,
    const MlirValue value,
    const MlirType type,
    const MlirLocation location);

  MlirOperation mlirGoCreatePtrToIntOperation(
    const MlirContext context,
    const MlirValue value,
    const MlirType type,
    const MlirLocation location);

  MlirOperation mlirGoCreateFloatTruncateOperation(
    const MlirContext context,
    const MlirValue value,
    const MlirType type,
    const MlirLocation location);

  MlirOperation mlirGoCreateIntTruncateOperation(
    const MlirContext context,
    const MlirValue value,
    const MlirType type,
    const MlirLocation location);

  MlirOperation mlirGoCreateFloatExtendOperation(
    const MlirContext context,
    const MlirValue value,
    const MlirType type,
    const MlirLocation location);

  MlirOperation mlirGoCreateSignedExtendOperation(
    const MlirContext context,
    const MlirValue value,
    const MlirType type,
    const MlirLocation location);

  MlirOperation mlirGoCreateZeroExtendOperation(
    const MlirContext context,
    const MlirValue value,
    const MlirType type,
    const MlirLocation location);

  MlirOperation mlirGoCreateFloatToUnsignedIntOperation(
    const MlirContext context,
    const MlirValue value,
    const MlirType type,
    const MlirLocation location);

  MlirOperation mlirGoCreateFloatToSignedIntOperation(
    const MlirContext context,
    const MlirValue value,
    const MlirType type,
    const MlirLocation location);

  MlirOperation mlirGoCreateUnsignedIntToFloatOperation(
    const MlirContext context,
    const MlirValue value,
    const MlirType type,
    const MlirLocation location);

  MlirOperation mlirGoCreateSignedIntToFloatOperation(
    const MlirContext context,
    const MlirValue value,
    const MlirType type,
    const MlirLocation location);

  MlirOperation mlirGoCreateFunctionToPointerOperation(
    const MlirContext context,
    const MlirValue value,
    const MlirType type,
    const MlirLocation location);

  MlirOperation mlirGoCreatePointerToFunctionOperation(
    const MlirContext context,
    const MlirValue value,
    const MlirType type,
    const MlirLocation location);

  MlirOperation mlirGoCreateChangeInterfaceOperation(
    const MlirContext context,
    const MlirValue value,
    const MlirType type,
    const MlirLocation location);

  MlirOperation mlirGoCreateTypeAssertOperation(
    const MlirContext context,
    const MlirValue value,
    const int nResults,
    const MlirType* results,
    const MlirLocation location);

  MlirOperation mlirGoCreateStringToSliceOperation(
    const MlirContext context,
    const MlirValue value,
    const MlirType type,
    const MlirLocation location);

  MlirOperation mlirGoCreateSliceToStringOperation(
    const MlirContext context,
    const MlirValue value,
    const MlirType type,
    const MlirLocation location);

  //===----------------------------------------------------------------------===//
  // Function Operations
  //===----------------------------------------------------------------------===//

  MlirOperation mlirGoGetFunction(
    const MlirContext context,
    const MlirStringRef symbol,
    const MlirType type,
    const MlirLocation location);

  //===----------------------------------------------------------------------===//
  // Builtin Operations
  //===----------------------------------------------------------------------===//

  MlirOperation
  mlirGoCreatePanicOperation(const MlirContext context, const MlirValue value, const MlirLocation location);

  MlirOperation
  mlirGoCreateRecoverOperation(const MlirContext context, const MlirType type, const MlirLocation location);

  //===----------------------------------------------------------------------===//
  // Atomic Operations
  //===----------------------------------------------------------------------===//

  MlirOperation mlirGoCreateAtomicAddIOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirValue addr,
    const MlirValue delta,
    const MlirLocation location);

  MlirOperation mlirGoCreateAtomicCompareAndSwapOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirValue addr,
    const MlirValue old,
    const MlirValue value,
    const MlirLocation location);

  MlirOperation mlirGoCreateAtomicSwapOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirValue addr,
    const MlirValue value,
    const MlirLocation location);

  //===----------------------------------------------------------------------===//
  // Control Flow Operations
  //===----------------------------------------------------------------------===//

  MlirOperation mlirGoCreateBranchOperation(
    const MlirContext context,
    const MlirBlock dest,
    const int nDestOperands,
    const MlirValue* destOperands,
    const MlirLocation location);

  MlirOperation mlirGoCreateCondBranchOperation(
    const MlirContext context,
    const MlirValue condition,
    const MlirBlock trueDest,
    const int nTrueDestOperands,
    const MlirValue* trueDestOperands,
    const MlirBlock falseDest,
    const int nFalseDestOperands,
    const MlirValue* falseDestOperands,
    const MlirLocation location);

  MlirOperation mlirGoCreateReturnOperation(
    const MlirContext context,
    const int nOperands,
    const MlirValue* operands,
    const MlirLocation location);

  //===----------------------------------------------------------------------===//
  // Call Operations
  //===----------------------------------------------------------------------===//

  MlirOperation mlirGoCreateCallOperation(
    const MlirContext context,
    const MlirStringRef callee,
    const int nResultTypes,
    const MlirType* resultTypes,
    const int nOperands,
    const MlirValue* operands,
    const MlirLocation location);

  MlirOperation mlirGoCreateClosureCallOperation(
    const MlirContext context,
    const MlirAttribute signature,
    const MlirValue callee,
    const int nResultTypes,
    const MlirType* resultTypes,
    const int nOperands,
    const MlirValue* operands,
    const MlirLocation location);

  MlirOperation mlirGoCreateCallIndirectOperation(
    const MlirContext context,
    const MlirValue callee,
    const int nResultTypes,
    const MlirType* resultTypes,
    const int nOperands,
    const MlirValue* operands,
    const MlirLocation location);

  MlirOperation mlirGoCreateDeferOperation1(
    const MlirContext context,
    const MlirStringRef sym_name,
    const int nArgs,
    const MlirValue* args,
    const MlirLocation location);

  MlirOperation mlirGoCreateDeferOperation2(
    const MlirContext context,
    const MlirAttribute sym_name,
    const int nArgs,
    const MlirValue* args,
    const MlirLocation location);

  MlirOperation mlirGoCreateDeferOperation3(
    const MlirContext context,
    const MlirAttribute signature,
    const MlirValue callee_value,
    const int nArgs,
    const MlirValue* args,
    const MlirLocation location);

  MlirOperation mlirGoCreateDeferOperation4(
    const MlirContext context,
    const MlirValue iface_value,
    const MlirStringRef method_name,
    const int nArgs,
    const MlirValue* args,
    const MlirLocation location);

  MlirOperation mlirGoCreateDeferOperation5(
    const MlirContext context,
    const MlirValue iface_value,
    const MlirAttribute method_name,
    const int nArgs,
    const MlirValue* args,
    const MlirLocation location);

  MlirOperation mlirGoCreateGoOperation1(
    const MlirContext context,
    const MlirStringRef sym_name,
    const int nArgs,
    const MlirValue* args,
    const MlirLocation location);

  MlirOperation mlirGoCreateGoOperation2(
    const MlirContext context,
    const MlirAttribute sym_name,
    const int nArgs,
    const MlirValue* args,
    const MlirLocation location);

  MlirOperation mlirGoCreateGoOperation3(
    const MlirContext context,
    const MlirAttribute signature,
    const MlirValue callee_value,
    const int nArgs,
    const MlirValue* args,
    const MlirLocation location);

  MlirOperation mlirGoCreateGoOperation4(
    const MlirContext context,
    const MlirValue iface_value,
    const MlirStringRef method_name,
    const int nArgs,
    const MlirValue* args,
    const MlirLocation location);

  MlirOperation mlirGoCreateGoOperation5(
    const MlirContext context,
    const MlirValue iface_value,
    const MlirAttribute method_name,
    const int nArgs,
    const MlirValue* args,
    const MlirLocation location);

  MlirOperation mlirGoCreateInterfaceCall(
    const MlirContext context,
    const MlirStringRef callee,
    const int nResultTypes,
    const MlirType* resultTypes,
    const MlirValue value,
    const int nArgs,
    const MlirValue* args,
    const MlirLocation location);

  MlirOperation mlirGoCreateBuiltInCallOperation(
    const MlirContext context,
    const MlirStringRef identifier,
    const int nResultTypes,
    const MlirType* resultTypes,
    const int nOperands,
    const MlirValue* operands,
    const MlirLocation location);

  //===----------------------------------------------------------------------===//
  // Value Operations
  //===----------------------------------------------------------------------===//

  MlirOperation
  mlirGoCreateZeroOperation(const MlirContext context, const MlirType type, const MlirLocation location);

  MlirOperation mlirGoCreateComplexOperation(
    const MlirContext context,
    const MlirType type,
    const MlirValue real,
    const MlirValue imag,
    const MlirLocation location);

  MlirOperation mlirGoCreateImagOperation(
    const MlirContext context,
    const MlirType type,
    const MlirValue value,
    const MlirLocation location);

  MlirOperation mlirGoCreateRealOperation(
    const MlirContext context,
    const MlirType type,
    const MlirValue value,
    const MlirLocation location);

  MlirOperation mlirGoCreateMakeMapOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirValue capacity,
    const MlirLocation location);

  MlirOperation mlirGoCreateMakeSliceOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirValue length,
    const MlirValue capacity,
    const MlirLocation location);

  MlirOperation mlirGoCreateMakeInterfaceOperation(
    const MlirContext context,
    const MlirType resultType,
    const MlirType type,
    const MlirValue value,
    const MlirLocation location);

  //===----------------------------------------------------------------------===//
  // Channel Operations
  //===----------------------------------------------------------------------===//

  MlirOperation mlirGoCreateChanRecvOp(
    const MlirContext context,
    const int nResultTypes,
    const MlirType* resultTypes,
    const MlirValue channel,
    const MlirLocation location);

  MlirOperation mlirGoCreateChanSendOp(
    const MlirContext context,
    const MlirValue channel,
    const MlirValue value,

    const MlirLocation location);

  MlirOperation mlirGoCreateChanRangeOp(
    const MlirContext context,
    const MlirValue channel,
    const MlirBlock bodyDest,
    const MlirBlock exitDest,
    const MlirLocation location);

  MlirOperation mlirGoCreateChanSelectOp(
    const MlirContext context,
    const bool hasDefault,
    const MlirAttribute send,
    const int nChans,
    const MlirValue* chans,
    const MlirBlock defaultDest,
    const MlirBlock exitDest,
    const int nCases,
    const MlirBlock* cases,
    const MlirLocation location);

#ifdef __cplusplus
}
#endif

#endif // GO_C_OPERATIONS_H