#ifndef GO_C_TYPES_H
#define GO_C_TYPES_H

#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/IR.h>
#include <mlir-c/Support.h>

#include "Go-c/mlir/Enums.h"

#ifdef __cplusplus
extern "C" {
#endif

MlirType mlirGoCreateNamedType(MlirType underlying, MlirStringRef name);

MlirType mlirGoGetUnderlyingType(MlirType type);

MlirType mlirGoGetBaseType(MlirType type);

bool mlirGoTypeIsAPointer(MlirType type);

intptr_t mlirGoGetTypeSizeInBytes(MlirType type, MlirModule module);

intptr_t mlirGoGetTypeSizeInBits(MlirType type, MlirModule module);

intptr_t mlirGoGetTypePreferredAlignmentInBytes(MlirType type, MlirModule module);

MlirType mlirGoCreateArrayType(MlirType elementType, intptr_t length);

MlirType mlirGoCreateChanType(MlirType elementType, enum mlirGoChanDirection direction);

MlirType mlirGoCreateInterfaceType(MlirContext context, intptr_t nMethodNames, MlirStringRef *methodNames,
                                   intptr_t nMethods, MlirType *methods);

MlirType mlirGoCreateNamedInterfaceType(MlirContext context, MlirStringRef name);

void mlirGoSetNamedInterfaceMethods(MlirContext context, MlirType interface, intptr_t nMethodNames,
                                    MlirStringRef *methodNames, intptr_t nMethods, MlirType *methods);

MlirType mlirGoCreateMapType(MlirType keyType, MlirType valueType);

MlirType mlirGoCreatePointerType(MlirType elementType);

MlirType mlirGoPointerTypeGetElementType(MlirType type);

MlirType mlirGoCreateUnsafePointerType(MlirContext context);

MlirType mlirGoCreateSliceType(MlirType elementType);

MlirType mlirGoCreateStringType(MlirContext context);

MlirType mlirGoCreateLiteralStructType(MlirContext context, intptr_t nFields, MlirType *fields);

MlirType mlirGoCreateNamedStructType(MlirContext context, MlirStringRef name);

void mlirGoSetStructTypeBody(MlirType type, intptr_t nFields, MlirType *fields);

MlirType mlirGoStructTypeGetFieldType(MlirType type, int index);

MlirType mlirGoCreateIntType(MlirContext ctx);

MlirType mlirGoCreateUintType(MlirContext ctx);

MlirType mlirGoCreateUintptrType(MlirContext ctx);

bool mlirGoTypeIsAIntType(MlirType T);

bool mlirGoTypeIsAUintType(MlirType T);

bool mlirGoTypeIsAUintptrType(MlirType T);

#ifdef __cplusplus
}
#endif

#endif // GO_C_TYPES_H
