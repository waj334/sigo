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

  MlirType mlirGoCreateUntypedType(MlirContext context, enum mlirGoBasicType basicType);

  enum mlirGoBasicType mlirGoUntypedTypeGetHBasicKind(MlirType type);

  MlirType mlirGoCreateNamedType(MlirType underlying, MlirStringRef name, MlirAttribute methods);

  MlirType mlirGoGetUnderlyingType(MlirType type);

  MlirType mlirGoGetBaseType(MlirType type);

  bool mlirGoTypeIsAPointer(MlirType type);

  MlirType mlirGoCreateArrayType(MlirType elementType, int length);

  MlirType mlirGoCreateChanType(MlirType elementType, enum mlirGoChanDirection direction);

  MlirType mlirGoCreateInterfaceType(
    MlirContext context,
    int nMethodNames,
    MlirStringRef* methodNames,
    int nMethods,
    MlirType* methods);

  MlirType mlirGoCreateNamedInterfaceType(MlirContext context, MlirStringRef name);

  void mlirGoSetNamedInterfaceMethods(
    MlirContext context,
    MlirType interface,
    int nMethodNames,
    MlirStringRef* methodNames,
    int nMethods,
    MlirType* methods);

  MlirType mlirGoCreateMapType(MlirType keyType, MlirType valueType);

  MlirType mlirGoCreatePointerType(MlirType elementType);

  MlirType mlirGoPointerTypeGetElementType(MlirType type);

  MlirType mlirGoCreateUnsafePointerType(MlirContext context);

  MlirType mlirGoCreateSliceType(MlirType elementType);

  MlirType mlirGoCreateStringType(MlirContext context);

  MlirType mlirGoCreateBasicStructType(MlirContext context, int nFields, MlirType* fields);

  MlirType mlirGoCreateLiteralStructType(
    MlirContext context,
    int nNames,
    MlirAttribute* names,
    int nFields,
    MlirType* fields,
    int nTags,
    MlirAttribute* tags);

  MlirType mlirGoCreateNamedStructType(MlirContext context, MlirStringRef name);

  void mlirGoSetStructTypeBody(
    MlirType type,
    int nNames,
    MlirAttribute* names,
    int nFields,
    MlirType* fields,
    int nTags,
    MlirAttribute* tags);

  MlirType mlirGoCreateBooleanType(MlirContext ctx);

  bool mlirGoTypeIsBoolean(MlirType type);

  MlirType mlirGoCreateSignedIntType(MlirContext ctx, int width);

  MlirType mlirGoStructTypeGetFieldType(MlirType type, int index);

  MlirType mlirGoCreateUnsignedIntType(MlirContext ctx, int width);

  MlirType mlirGoCreateUintptrType(MlirContext ctx);

  bool mlirGoTypeIsInteger(MlirType type);

  bool mlirGoTypeIsUntyped(MlirType type);

  bool mlirGoIntegerTypeIsSigned(MlirType type);

  bool mlirGoIntegerTypeIsUnsigned(MlirType type);

  bool mlirGoIntegerTypeIsUintptr(MlirType type);

  int mlirGoIntegerTypeGetWidth(MlirType type);

  MlirType mlirGoCreateFunctionType(
    MlirContext ctx,
    MlirType* receiver,
    int nInputs,
    MlirType* inputs,
    int nResults,
    MlirType* results);

  bool mlirGoTypeIsAFunctionType(MlirType type);

  bool mlirGoFunctionTypeHasReceiver(MlirType type);

  MlirType mlirGoFunctionTypeGetReceiver(MlirType type);

  int mlirGoFunctionTypeGetNumInputs(MlirType type);

  MlirType mlirGoFunctionTypeGetInput(MlirType type, int index);

  int mlirGoFunctionTypeGetNumResults(MlirType type);

  MlirType mlirGoFunctionTypeGetResult(MlirType type, int index);

#ifdef __cplusplus
}
#endif

#endif // GO_C_TYPES_H
