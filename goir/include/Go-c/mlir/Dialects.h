#ifndef GO_C_DIALECTS_H
#define GO_C_DIALECTS_H

#include <llvm-c/Target.h>

#include <mlir-c/IR.h>

#include "Enums.h"

#ifdef __cplusplus
extern "C"
{
#endif

  MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Go, go);

  void mlirGoInitializeContext(MlirContext context);

  MlirStringRef mlirModuleDump(MlirModule module);

  bool mlirModuleDumpToFile(MlirModule module, MlirStringRef fname);

  void mlirStringRefDestroy(MlirStringRef* ref);

  int mlirTypeHash(MlirType type);

  MlirAttribute mlirGoCreateTypeMetadata(MlirType type, MlirAttribute dict);

  MlirStringRef mlirGoGetTypeInfoSymbol(MlirType type, MlirStringRef prefix);

  MlirOperation mlirCreateUnrealizedConversionCastOp(
    MlirContext context,
    MlirType type,
    MlirValue value,
    MlirLocation location);

  void mlirGoBindRuntimeType(MlirModule module, MlirStringRef mnemonic, MlirType runtimeType);

  void mlirGoBindRuntimeTypeToType(MlirModule module, MlirType primitiveType, MlirType runtimeType);

  void mlirGoSetTargetDataLayout(MlirModule module, LLVMTargetDataRef layout);

  void mlirGoSetTargetTriple(MlirModule module, MlirStringRef triple);

  MlirLogicalResult mlirCanonicalizeModule(MlirModule module);

  MlirLogicalResult
  mlirGoOptimizeModule(MlirModule module, MlirStringRef name, MlirStringRef outputDir, bool debug);

  MlirAttribute mlirGetLLVMLinkageAttr(MlirContext context, MlirStringRef linkage);

  void mlirInitModuleTranslation(MlirContext context);

  LLVMModuleRef
  mlirTranslateModuleToLLVMIR(MlirModule module, LLVMContextRef llvmContext, MlirStringRef name);

  MlirAttribute mlirGoCreateTypeMetadataEntryAttr(MlirType type, MlirAttribute dict);

  MlirAttribute mlirGoCreateTypeMetadataDictionaryAttr(
    MlirContext context,
    int nEntries,
    MlirAttribute* entries);

  MlirBlock mlirRegionGetLastBlock(MlirRegion region);

  MlirLogicalResult mlirVerifyModule(MlirModule module);

  MlirAttribute
  mlirGoCreateComplexNumberAttr(MlirContext context, MlirType type, double real, double imag);

  bool mlirOperationHasNoMemoryEffect(MlirOperation op);

  MlirOperation mlirValueGetDefiningOperation(MlirValue value);

  MlirBlock
  mlirBlockCreate2(int nArgs, MlirType* args, int nLocations, MlirLocation* locations);

  MlirAttribute mlirDistinctAttrGet(MlirAttribute attr);

  void mlirGoBlockDumpTail(MlirBlock block, int count);

  void mlirGoMoveBlockAfter(const MlirBlock block, const MlirBlock after);

  enum MlirGoAsmConstraintDirection
  {
    In,
    Out,
    InOut
  };

  MlirAttribute mlirGoCreateAsmConstraintAttr(
    MlirContext context,
    MlirStringRef registerClass,
    enum MlirGoAsmConstraintDirection direction,
    MlirStringRef alias,
    int operandIndex,
    bool reserve);

#ifdef __cplusplus
}
#endif

#endif // GO_C_DIALECTS_H
