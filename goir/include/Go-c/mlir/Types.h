#ifndef GO_C_TYPES_H
#define GO_C_TYPES_H

#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/IR.h>
#include <mlir-c/Support.h>

#include "Go-c/mlir/Enums.h"

#ifdef __cplusplus
extern "C"
{
#endif

  MlirType mlirGoCreateNamedType(MlirType underlying, MlirStringRef name, MlirAttribute methods);

  MlirType mlirGoGetUnderlyingType(MlirType type);

  MlirType mlirGoGetBaseType(MlirType type);

  bool mlirGoTypeIsAPointer(MlirType type);

  intptr_t mlirGoGetTypeSizeInBytes(MlirType type, MlirModule module);

  intptr_t mlirGoGetTypeSizeInBits(MlirType type, MlirModule module);

  intptr_t mlirGoGetTypePreferredAlignmentInBytes(MlirType type, MlirModule module);

  MlirType mlirGoCreateArrayType(MlirType elementType, intptr_t length);

  MlirType mlirGoCreateChanType(MlirType elementType, enum mlirGoChanDirection direction);

  MlirType mlirGoCreateInterfaceType(
    MlirContext context,
    intptr_t nMethodNames,
    MlirStringRef* methodNames,
    intptr_t nMethods,
    MlirType* methods);

  MlirType mlirGoCreateNamedInterfaceType(MlirContext context, MlirStringRef name);

  void mlirGoSetNamedInterfaceMethods(
    MlirContext context,
    MlirType interface,
    intptr_t nMethodNames,
    MlirStringRef* methodNames,
    intptr_t nMethods,
    MlirType* methods);

  MlirType mlirGoCreateMapType(MlirType keyType, MlirType valueType);

  MlirType mlirGoCreatePointerType(MlirType elementType);

  MlirType mlirGoPointerTypeGetElementType(MlirType type);

  MlirType mlirGoCreateUnsafePointerType(MlirContext context);

  MlirType mlirGoCreateSliceType(MlirType elementType);

  MlirType mlirGoCreateStringType(MlirContext context);

  MlirType mlirGoCreateBasicStructType(MlirContext context, intptr_t nFields, MlirType* fields);

  MlirType mlirGoCreateLiteralStructType(
    MlirContext context,
    intptr_t nNames,
    MlirAttribute* names,
    intptr_t nFields,
    MlirType* fields,
    intptr_t nTags,
    MlirAttribute* tags);

  MlirType mlirGoCreateNamedStructType(MlirContext context, MlirStringRef name);

  void mlirGoSetStructTypeBody(
    MlirType type,
    intptr_t nNames,
    MlirAttribute* names,
    intptr_t nFields,
    MlirType* fields,
    intptr_t nTags,
    MlirAttribute* tags);

  MlirType mlirGoCreateBooleanType(MlirContext ctx);

  bool mlirGoTypeIsBoolean(MlirType type);

  MlirType mlirGoCreateSignedIntType(MlirContext ctx, intptr_t width);

  MlirType mlirGoStructTypeGetFieldType(MlirType type, int index);

  MlirType mlirGoCreateUnsignedIntType(MlirContext ctx, intptr_t width);

  MlirType mlirGoCreateUintptrType(MlirContext ctx);

  bool mlirGoTypeIsInteger(MlirType type);

  bool mlirGoIntegerTypeIsSigned(MlirType type);

  bool mlirGoIntegerTypeIsUnsigned(MlirType type);

  bool mlirGoIntegerTypeIsUintptr(MlirType type);

  int mlirGoIntegerTypeGetWidth(MlirType type);

  MlirType mlirGoCreateFunctionType(
    MlirContext ctx,
    MlirType* receiver,
    intptr_t nInputs,
    MlirType* inputs,
    intptr_t nResults,
    MlirType* results);

  bool mlirGoTypeIsAFunctionType(MlirType type);

  bool mlirGoFunctionTypeHasReceiver(MlirType type);

  MlirType mlirGoFunctionTypeGetReceiver(MlirType type);

  intptr_t mlirGoFunctionTypeGetNumInputs(MlirType type);

  MlirType mlirGoFunctionTypeGetInput(MlirType type, intptr_t index);

  intptr_t mlirGoFunctionTypeGetNumResults(MlirType type);

  MlirType mlirGoFunctionTypeGetResult(MlirType type, intptr_t index);

#ifdef __cplusplus
}
#endif

#endif // GO_C_TYPES_H
