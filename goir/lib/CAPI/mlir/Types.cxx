#include "Go-c/mlir/Types.h"

#include "Go/IR/GoTypes.h"

#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Support.h>
#include <mlir/IR/Types.h>

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#include "Go/Util.h"

MlirType mlirGoCreateNamedType(MlirType underlying, MlirStringRef package, MlirStringRef name, MlirAttribute* methods) {
  const auto _underlying = unwrap(underlying);
  const auto ctx = _underlying.getContext();

  const auto _name = mlir::StringAttr::get(ctx, unwrap(name));
  const auto _package = mlir::StringAttr::get(ctx, unwrap(package));

  std::optional<mlir::ArrayAttr> _methods;
  if (methods != nullptr) {
    _methods = mlir::cast<mlir::ArrayAttr>(unwrap(*methods));
  }
  return wrap(mlir::go::NamedType::get(ctx, _underlying, _name, _package, _methods));
}

MlirType mlirGoGetUnderlyingType(MlirType type) {
  const auto _type = unwrap(type);
  return wrap(mlir::go::underlyingType(_type));
}

MlirType mlirGoGetBaseType(MlirType type) {
  const auto _type = unwrap(type);
  return wrap(mlir::go::baseType(_type));
}

bool mlirGoTypeIsAPointer(MlirType type) {
  auto _type = unwrap(type);
  return mlir::go::isa<mlir::go::PointerType>(_type);
}

intptr_t mlirGoGetTypeSizeInBytes(MlirType type, MlirModule module) {
    auto _type = unwrap(type);
    auto _module = unwrap(module);

    mlir::DataLayout dataLayout(_module);
    const auto dataLayoutSpec = _module.getDataLayoutSpec();
    return mlir::go::getDefaultTypeSize(_type, dataLayout, dataLayoutSpec.getEntries());
}

intptr_t mlirGoGetTypeSizeInBits(MlirType type, MlirModule module) {
    auto _type = unwrap(type);
    auto _module = unwrap(module);

    mlir::DataLayout dataLayout(_module);
    const auto dataLayoutSpec = _module.getDataLayoutSpec();
    return mlir::go::getDefaultTypeSizeInBits(_type, dataLayout, dataLayoutSpec.getEntries());
}

MlirType mlirGoCreateArrayType(MlirType elementType, intptr_t length) {
  auto _elementType = unwrap(elementType);
  return wrap(mlir::go::ArrayType::get(_elementType.getContext(), _elementType, length));
}

MlirType mlirGoCreateChanType(MlirType elementType, enum mlirGoChanDirection direction) {
  ::mlir::go::ChanDirection dir;
  auto _elementType = unwrap(elementType);
  switch (direction) {
  case mlirGoChanDirection_SendRecv:
    dir = ::mlir::go::ChanDirection::SendRecv;
    break;
  case mlirGoChanDirection_SendOnly:
    dir = ::mlir::go::ChanDirection::SendOnly;
    break;
  case mlirGoChanDirection_RecvOnly:
    dir = ::mlir::go::ChanDirection::RecvOnly;
    break;
  default:
    assert(false && "unreachable");
  }
  return wrap(mlir::go::ChanType::get(_elementType.getContext(), _elementType, dir));
}

MlirType mlirGoCreateInterfaceType(MlirContext context, intptr_t nMethodNames, MlirStringRef *methodNames,
                                   intptr_t nMethods, MlirType *methods) {
  auto _context = unwrap(context);

  ::llvm::SmallVector<::mlir::StringRef> _methodNames;
  (void)unwrapList(nMethodNames, methodNames, _methodNames);

  ::llvm::SmallVector<::mlir::Type> _methods;
  (void)unwrapList(nMethods, methods, _methods);

  // Assert that all methods are functions
  for (const auto &method : _methods) {
    assert(method.isa<::mlir::FunctionType>() && "type must be a function");
  }

  // Assert that all methods have a corresponding name
  assert(_methodNames.size() == _methods.size());

  // Populate the method map
  mlir::go::detail::InterfaceTypeStorage::FunctionMap _methodsMap;
  for (intptr_t i = 0; i < nMethodNames; i++) {
    _methodsMap[_methodNames[i].str()] = mlir::cast<mlir::FunctionType>(_methods[i]);
  }

  // Return the interface type
  return wrap(mlir::go::InterfaceType::get(_context, _methodsMap));
}

MlirType mlirGoCreateNamedInterfaceType(MlirContext context, MlirStringRef name) {
  auto _context = unwrap(context);
  auto _name = unwrap(name);

  assert(!_name.empty());
  return wrap(::mlir::go::InterfaceType::getNamed(_context, _name.str()));
}

void mlirGoSetNamedInterfaceMethods(MlirContext context, MlirType interface, intptr_t nMethodNames,
                                    MlirStringRef *methodNames, intptr_t nMethods, MlirType *methods) {
  auto _interface = mlir::cast<mlir::go::InterfaceType>(unwrap(interface));

  ::llvm::SmallVector<::mlir::StringRef> _methodNames;
  (void)unwrapList(nMethodNames, methodNames, _methodNames);

  ::llvm::SmallVector<::mlir::Type> _methods;
  (void)unwrapList(nMethods, methods, _methods);

  // Assert that all methods are functions
  for (const auto &method : _methods) {
    assert(method.isa<::mlir::FunctionType>() && "type must be a function");
  }

  // Assert that all methods have a corresponding name
  assert(_methodNames.size() == _methods.size());

  // Populate the method map
  mlir::go::detail::InterfaceTypeStorage::FunctionMap _methodsMap;
  for (intptr_t i = 0; i < nMethodNames; i++) {
    _methodsMap[_methodNames[i].str()] = mlir::cast<mlir::FunctionType>(_methods[i]);
  }

  _interface.setMethods(_methodsMap);
}

MlirType mlirGoCreateMapType(MlirType keyType, MlirType valueType) {
  auto _keyType = unwrap(keyType);
  auto _valueType = unwrap(valueType);
  assert((_keyType.getContext() == _valueType.getContext()) &&
         "key and value types must originate from the same context");

  return wrap(mlir::go::MapType::get(_keyType.getContext(), _keyType, _valueType));
}

MlirType mlirGoCreatePointerType(MlirType elementType) {
  auto _elementType = unwrap(elementType);
  return wrap(mlir::go::PointerType::get(_elementType.getContext(), _elementType));
}

MlirType mlirGoPointerTypeGetElementType(MlirType type) {
  auto _type = mlir::cast<mlir::go::PointerType>(unwrap(type));
  if (_type.getElementType()) {
    return wrap(*_type.getElementType());
  }
  return wrap(mlir::Type());
}

MlirType mlirGoCreateUnsafePointerType(MlirContext context) {
  auto _context = unwrap(context);
  return wrap(mlir::go::PointerType::get(_context, {}));
}

MlirType mlirGoCreateSliceType(MlirType elementType) {
  auto _elementType = unwrap(elementType);
  return wrap(mlir::go::SliceType::get(_elementType.getContext(), _elementType));
}

MlirType mlirGoCreateStringType(MlirContext context) { return wrap(::mlir::go::StringType::get(unwrap(context))); }

MlirType mlirGoCreateLiteralStructType(MlirContext context, intptr_t nFields, MlirType *fields) {
  auto _context = unwrap(context);
  mlir::DictionaryAttr _methods;

  ::llvm::SmallVector<::mlir::Type> tmp;
  (void)unwrapList(nFields, fields, tmp);

  std::vector<::mlir::Type> _fields(tmp.begin(), tmp.end());
  return wrap(mlir::LLVM::LLVMStructType::getLiteral(_context, _fields));
}

MlirType mlirGoCreateNamedStructType(MlirContext context, MlirStringRef name) {
  auto _name = unwrap(name);
  auto _context = unwrap(context);
  mlir::DictionaryAttr _methods;
  return wrap(mlir::LLVM::LLVMStructType::getIdentified(_context, _name));
}

void mlirGoSetStructTypeBody(MlirType type, intptr_t nFields, MlirType *fields) {
  std::vector<mlir::Type> body;
  auto _type = unwrap(type);

  ::llvm::SmallVector<::mlir::Type> tmp;
  (void)unwrapList(nFields, fields, tmp);

  std::vector<::mlir::Type> _fields(tmp.begin(), tmp.end());
  (void)mlir::dyn_cast<mlir::LLVM::LLVMStructType>(_type).setBody(_fields, false);
}

MlirType mlirGoStructTypeGetFieldType(MlirType type, int index) {
  auto _type = mlir::cast<mlir::LLVM::LLVMStructType>(unwrap(type));
  return wrap(_type.getBody()[index]);
}

MlirType mlirGoCreateIntType(MlirContext ctx) {
  const auto _ctx = unwrap(ctx);
  return wrap(mlir::go::IntType::get(_ctx));
}

MlirType mlirGoCreateUintType(MlirContext ctx) {
  const auto _ctx = unwrap(ctx);
  return wrap(mlir::go::UintType::get(_ctx));
}

MlirType mlirGoCreateUintptrType(MlirContext ctx) {
  const auto _ctx = unwrap(ctx);
  return wrap(mlir::go::UintptrType::get(_ctx));
}

bool mlirGoTypeIsAIntType(MlirType T) {
  auto _T = unwrap(T);
  return mlir::isa<mlir::go::IntType>(_T);
}

bool mlirGoTypeIsAUintType(MlirType T) {
  auto _T = unwrap(T);
  return mlir::isa<mlir::go::UintType>(_T);
}

bool mlirGoTypeIsAUintptrType(MlirType T) {
  auto _T = unwrap(T);
  return mlir::isa<mlir::go::UintptrType>(_T);
}