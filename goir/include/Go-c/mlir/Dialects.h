#ifndef GO_C_DIALECTS_H
#define GO_C_DIALECTS_H

#include "Enums.h"

#include <llvm-c/Target.h>
#include <mlir-c/IR.h>

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Go, go);

MlirStringRef mlirModuleDump(MlirModule module);

bool mlirModuleDumpToFile(MlirModule module, MlirStringRef fname);

void mlirStringRefDestroy(MlirStringRef *ref);

intptr_t mlirTypeHash(MlirType type);

MlirAttribute mlirGoCreateTypeMetadata(MlirType type, MlirAttribute dict);

MlirStringRef mlirGoGetTypeInfoSymbol(MlirType type, MlirStringRef prefix);

MlirOperation mlirCreateUnrealizedConversionCastOp(MlirContext context, MlirType type, MlirValue value,
                                                   MlirLocation location);

void mlirGoBindRuntimeType(MlirModule module, MlirStringRef mnemonic, MlirType runtimeType);

void mlirGoSetTargetDataLayout(MlirModule module, LLVMTargetDataRef layout);

void mlirGoSetTargetTriple(MlirModule module, MlirStringRef triple);

MlirLogicalResult mlirCanonicalizeModule(MlirModule module);

MlirLogicalResult mlirGoOptimizeModule(MlirModule module, MlirStringRef name, MlirStringRef outputDir, bool debug);

MlirAttribute mlirGetLLVMLinkageAttr(MlirContext context, MlirStringRef linkage);

void mlirInitModuleTranslation(MlirContext context);

LLVMModuleRef mlirTranslateModuleToLLVMIR(MlirModule module, LLVMContextRef llvmContext, MlirStringRef name);

MlirAttribute mlirGoCreateTypeMetadataEntryAttr(MlirType type, MlirAttribute dict);

MlirAttribute mlirGoCreateTypeMetadataDictionaryAttr(MlirContext context, intptr_t nEntries, MlirAttribute *entries);

MlirBlock mlirRegionGetLastBlock(MlirRegion region);

MlirLogicalResult mlirVerifyModule(MlirModule module);

MlirAttribute mlirGoCreateComplexNumberAttr(MlirContext context, MlirType type, double real, double imag);

bool mlirOperationHasNoMemoryEffect(MlirOperation op);

MlirOperation mlirValueGetDefiningOperation(MlirValue value);

MlirBlock mlirBlockCreate2(intptr_t nArgs, MlirType *args, intptr_t nLocations, MlirLocation *locations);

MlirAttribute mlirDistinctAttrGet(MlirAttribute attr);

void mlirGoBlockDumpTail(MlirBlock block, intptr_t count);

#ifdef __cplusplus
}
#endif

#endif // GO_C_DIALECTS_H
