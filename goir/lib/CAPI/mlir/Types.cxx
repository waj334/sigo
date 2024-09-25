#include "Go-c/mlir/Types.h"

#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Support.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/Types.h>

#include "Go/IR/GoTypes.h"
#include "Go/Util.h"

MlirType mlirGoCreateNamedType(MlirType underlying, MlirStringRef name, MlirAttribute methods)
{
  const auto _underlying = unwrap(underlying);
  const auto ctx = _underlying.getContext();
  const auto _name = mlir::StringAttr::get(ctx, unwrap(name));
  const auto _methods = mlir::cast<mlir::ArrayAttr>(unwrap(methods));
  for (const auto& attr : _methods)
  {
    assert(mlir::isa<mlir::FlatSymbolRefAttr>(attr));
  }
  return wrap(mlir::go::NamedType::get(ctx, _underlying, _name, _methods));
}

MlirType mlirGoGetUnderlyingType(MlirType type)
{
  const auto _type = unwrap(type);
  return wrap(mlir::go::underlyingType(_type));
}

MlirType mlirGoGetBaseType(MlirType type)
{
  const auto _type = unwrap(type);
  return wrap(mlir::go::baseType(_type));
}

bool mlirGoTypeIsAPointer(MlirType type)
{
  auto _type = unwrap(type);
  return mlir::go::isa<mlir::go::PointerType>(_type);
}

intptr_t mlirGoGetTypeSizeInBytes(MlirType type, MlirModule module)
{
  auto _type = unwrap(type);
  auto _module = unwrap(module);

  mlir::DataLayout dataLayout(_module);
  const auto dataLayoutSpec = _module.getDataLayoutSpec();
  return mlir::go::getDefaultTypeSize(_type, dataLayout, dataLayoutSpec.getEntries());
}

intptr_t mlirGoGetTypeSizeInBits(MlirType type, MlirModule module)
{
  auto _type = unwrap(type);
  auto _module = unwrap(module);

  mlir::DataLayout dataLayout(_module);
  const auto dataLayoutSpec = _module.getDataLayoutSpec();
  return mlir::go::getDefaultTypeSizeInBits(_type, dataLayout, dataLayoutSpec.getEntries());
}

intptr_t mlirGoGetTypePreferredAlignmentInBytes(MlirType type, MlirModule module)
{
  auto _type = unwrap(type);
  auto _module = unwrap(module);

  mlir::DataLayout dataLayout(_module);
  const auto dataLayoutSpec = _module.getDataLayoutSpec();
  return mlir::go::getDefaultPreferredAlignment(_type, dataLayout, dataLayoutSpec.getEntries());
}

MlirType mlirGoCreateArrayType(MlirType elementType, intptr_t length)
{
  auto _elementType = unwrap(elementType);
  return wrap(mlir::go::ArrayType::get(_elementType.getContext(), _elementType, length));
}

MlirType mlirGoCreateChanType(MlirType elementType, enum mlirGoChanDirection direction)
{
  ::mlir::go::ChanDirection dir;
  auto _elementType = unwrap(elementType);
  switch (direction)
  {
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

MlirType mlirGoCreateInterfaceType(
  MlirContext context,
  intptr_t nMethodNames,
  MlirStringRef* methodNames,
  intptr_t nMethods,
  MlirType* methods)
{
  auto _context = unwrap(context);

  ::llvm::SmallVector<::mlir::StringRef> _methodNames;
  (void)unwrapList(nMethodNames, methodNames, _methodNames);

  ::llvm::SmallVector<::mlir::Type> _methods;
  (void)unwrapList(nMethods, methods, _methods);

  // Assert that all methods are functions
  for (const auto& method : _methods)
  {
    assert(mlir::isa<mlir::go::FunctionType>(method) && "type must be a function");
  }

  // Assert that all methods have a corresponding name
  assert(_methodNames.size() == _methods.size());

  // Populate the method map
  mlir::go::detail::InterfaceTypeStorage::FunctionMap _methodsMap;
  for (intptr_t i = 0; i < nMethodNames; i++)
  {
    _methodsMap[_methodNames[i].str()] = mlir::cast<mlir::go::FunctionType>(_methods[i]);
  }

  // Return the interface type
  return wrap(mlir::go::InterfaceType::get(_context, _methodsMap));
}

MlirType mlirGoCreateNamedInterfaceType(MlirContext context, MlirStringRef name)
{
  auto _context = unwrap(context);
  auto _name = unwrap(name);

  assert(!_name.empty());
  return wrap(::mlir::go::InterfaceType::getNamed(_context, _name.str()));
}

void mlirGoSetNamedInterfaceMethods(
  MlirContext context,
  MlirType interface,
  intptr_t nMethodNames,
  MlirStringRef* methodNames,
  intptr_t nMethods,
  MlirType* methods)
{
  auto _interface = mlir::cast<mlir::go::InterfaceType>(unwrap(interface));

  ::llvm::SmallVector<::mlir::StringRef> _methodNames;
  (void)unwrapList(nMethodNames, methodNames, _methodNames);

  ::llvm::SmallVector<::mlir::Type> _methods;
  (void)unwrapList(nMethods, methods, _methods);

  // Assert that all methods are functions
  for (const auto& method : _methods)
  {
    assert(mlir::isa<mlir::go::FunctionType>(method) && "type must be a Go function");
  }

  // Assert that all methods have a corresponding name
  assert(_methodNames.size() == _methods.size());

  // Populate the method map
  mlir::go::detail::InterfaceTypeStorage::FunctionMap _methodsMap;
  for (intptr_t i = 0; i < nMethodNames; i++)
  {
    _methodsMap[_methodNames[i].str()] = mlir::cast<mlir::go::FunctionType>(_methods[i]);
  }

  _interface.setMethods(_methodsMap);
}

MlirType mlirGoCreateMapType(MlirType keyType, MlirType valueType)
{
  auto _keyType = unwrap(keyType);
  auto _valueType = unwrap(valueType);
  assert(
    (_keyType.getContext() == _valueType.getContext()) &&
    "key and value types must originate from the same context");

  return wrap(mlir::go::MapType::get(_keyType.getContext(), _keyType, _valueType));
}

MlirType mlirGoCreatePointerType(MlirType elementType)
{
  auto _elementType = unwrap(elementType);
  return wrap(mlir::go::PointerType::get(_elementType.getContext(), _elementType));
}

MlirType mlirGoPointerTypeGetElementType(MlirType type)
{
  auto _type = mlir::cast<mlir::go::PointerType>(unwrap(type));
  if (_type.getElementType())
  {
    return wrap(*_type.getElementType());
  }
  return wrap(mlir::Type());
}

MlirType mlirGoCreateUnsafePointerType(MlirContext context)
{
  auto _context = unwrap(context);
  return wrap(mlir::go::PointerType::get(_context, {}));
}

MlirType mlirGoCreateSliceType(MlirType elementType)
{
  auto _elementType = unwrap(elementType);
  return wrap(mlir::go::SliceType::get(_elementType.getContext(), _elementType));
}

MlirType mlirGoCreateStringType(MlirContext context)
{
  return wrap(::mlir::go::StringType::get(unwrap(context)));
}

MlirType mlirGoCreateBasicStructType(MlirContext context, intptr_t nFields, MlirType* fields)
{
  auto _context = unwrap(context);

  mlir::SmallVector<::mlir::Type> _fieldTypes;
  (void)unwrapList(nFields, fields, _fieldTypes);

  return wrap(mlir::go::GoStructType::getBasic(_context, _fieldTypes));
}

MlirType mlirGoCreateLiteralStructType(
  MlirContext context,
  intptr_t nNames,
  MlirAttribute* names,
  intptr_t nFields,
  MlirType* fields,
  intptr_t nTags,
  MlirAttribute* tags)
{
  auto _context = unwrap(context);

  mlir::SmallVector<mlir::StringAttr> _fieldNames;
  _fieldNames.reserve(nNames);

  mlir::SmallVector<mlir::StringAttr> _fieldTags;
  _fieldTags.reserve(nTags);

  mlir::SmallVector<::mlir::Attribute> tmp;

  // Unwrap field names.
  (void)unwrapList(nNames, names, tmp);
  for (auto attr : tmp)
  {
    if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr))
    {
      _fieldNames.push_back(strAttr);
    }
    else
    {
      assert(false && "unsupported attribute type");
    }
  }

  // Unwrap field tags.
  tmp.clear();
  (void)unwrapList(nTags, tags, tmp);
  for (auto attr : tmp)
  {
    if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr))
    {
      _fieldTags.push_back(strAttr);
    }
    else
    {
      assert(false && "unsupported attribute type");
    }
  }

  mlir::SmallVector<::mlir::Type> _fieldTypes;
  (void)unwrapList(nFields, fields, _fieldTypes);

  assert(_fieldNames.size() == _fieldTypes.size() && _fieldTags.size() == _fieldTypes.size());

  mlir::SmallVector<mlir::go::GoStructType::FieldTy> _fields;
  _fields.reserve(nFields);
  for (intptr_t i = 0; i < nFields; i++)
  {
    _fields.emplace_back(_fieldNames[i], _fieldTypes[i], _fieldTags[i]);
  }

  return wrap(mlir::go::GoStructType::getLiteral(_context, _fields));
}

MlirType mlirGoCreateNamedStructType(MlirContext context, MlirStringRef name)
{
  auto _name = unwrap(name);
  auto _context = unwrap(context);
  return wrap(mlir::go::GoStructType::get(_context, _name));
}

void mlirGoSetStructTypeBody(
  MlirType type,
  intptr_t nNames,
  MlirAttribute* names,
  intptr_t nFields,
  MlirType* fields,
  intptr_t nTags,
  MlirAttribute* tags)
{
  std::vector<mlir::Type> body;
  auto _type = unwrap(type);

  mlir::SmallVector<mlir::StringAttr> _fieldNames;
  _fieldNames.reserve(nNames);

  mlir::SmallVector<mlir::StringAttr> _fieldTags;
  _fieldTags.reserve(nTags);

  mlir::SmallVector<::mlir::Attribute> tmp;

  // Unwrap field names.
  (void)unwrapList(nNames, names, tmp);
  for (auto attr : tmp)
  {
    if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr))
    {
      _fieldNames.push_back(strAttr);
    }
    else
    {
      assert(false && "unsupported attribute type");
    }
  }

  // Unwrap field tags.
  tmp.clear();
  (void)unwrapList(nTags, tags, tmp);
  for (auto attr : tmp)
  {
    if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr))
    {
      _fieldTags.push_back(strAttr);
    }
    else
    {
      assert(false && "unsupported attribute type");
    }
  }

  mlir::SmallVector<::mlir::Type> _fieldTypes;
  (void)unwrapList(nFields, fields, _fieldTypes);

  assert(_fieldNames.size() == _fieldTypes.size() && _fieldTags.size() == _fieldTypes.size());

  mlir::SmallVector<mlir::go::GoStructType::FieldTy> _fields;
  _fieldNames.reserve(nFields);
  for (intptr_t i = 0; i < nFields; i++)
  {
    _fields.emplace_back(_fieldNames[i], _fieldTypes[i], _fieldTags[i]);
  }

  (void)mlir::dyn_cast<mlir::go::GoStructType>(_type).setFields(_fields);
}

MlirType mlirGoStructTypeGetFieldType(MlirType type, int index)
{
  auto _type = mlir::cast<mlir::go::GoStructType>(unwrap(type));
  return wrap(_type.getFieldType(index));
}

MlirType mlirGoCreateBooleanType(MlirContext ctx)
{
  auto _ctx = unwrap(ctx);
  return wrap(mlir::go::BooleanType::get(_ctx));
}

bool mlirGoTypeIsBoolean(MlirType type)
{
  return mlir::go::isa<mlir::go::BooleanType>(unwrap(type));
}

MlirType mlirGoCreateSignedIntType(MlirContext ctx, intptr_t width)
{
  auto _ctx = unwrap(ctx);
  if (width > 0)
  {
    return wrap(mlir::go::IntegerType::get(_ctx, mlir::go::IntegerType::Signed, width));
  }
  return wrap(mlir::go::IntegerType::get(_ctx, mlir::go::IntegerType::Signed));
}

MlirType mlirGoCreateUnsignedIntType(MlirContext ctx, intptr_t width)
{
  auto _ctx = unwrap(ctx);
  if (width > 0)
  {
    return wrap(mlir::go::IntegerType::get(_ctx, mlir::go::IntegerType::Unsigned, width));
  }
  return wrap(mlir::go::IntegerType::get(_ctx, mlir::go::IntegerType::Unsigned));
}

MlirType mlirGoCreateUintptrType(MlirContext ctx)
{
  auto _ctx = unwrap(ctx);
  return wrap(mlir::go::IntegerType::get(_ctx, mlir::go::IntegerType::Uintptr));
}

bool mlirGoTypeIsInteger(MlirType type)
{
  const auto _type = mlir::go::dyn_cast<mlir::go::IntegerType>(unwrap(type));
  if (!_type)
  {
    return false;
  }
  return true;
}

bool mlirGoIntegerTypeIsSigned(MlirType type)
{
  const auto _type = mlir::go::cast<mlir::go::IntegerType>(unwrap(type));
  return _type.isSigned();
}

bool mlirGoIntegerTypeIsUnsigned(MlirType type)
{
  const auto _type = mlir::go::cast<mlir::go::IntegerType>(unwrap(type));
  return _type.isUnsigned();
}

bool mlirGoIntegerTypeIsUintptr(MlirType type)
{
  const auto _type = mlir::go::cast<mlir::go::IntegerType>(unwrap(type));
  return _type.isUnsignedInteger();
}

int mlirGoIntegerTypeGetWidth(MlirType type)
{
  const auto _type = mlir::go::cast<mlir::go::IntegerType>(unwrap(type));
  const auto width = _type.getWidth();
  if (width.has_value())
  {
    return *width;
  }
  return 0;
}

MlirType mlirGoCreateFunctionType(
  MlirContext ctx,
  MlirType* receiver,
  intptr_t nInputs,
  MlirType* inputs,
  intptr_t nResults,
  MlirType* results)
{
  auto _ctx = unwrap(ctx);
  std::optional<mlir::Type> _receiver;
  if (receiver != nullptr)
  {
    _receiver = unwrap(*receiver);
  }

  mlir::SmallVector<mlir::Type> _inputs;
  (void)unwrapList(nInputs, inputs, _inputs);

  mlir::SmallVector<mlir::Type> _results;
  (void)unwrapList(nResults, results, _results);

  return wrap(mlir::go::FunctionType::get(_ctx, _inputs, _results, _receiver));
}

bool mlirGoFunctionTypeHasReceiver(MlirType type)
{
  const auto _type = mlir::go::cast<mlir::go::FunctionType>(unwrap(type));
  if (_type.getReceiver())
  {
    return true;
  }
  return false;
}

MlirType mlirGoFunctionTypeGetReceiver(MlirType type)
{
  const auto _type = mlir::go::cast<mlir::go::FunctionType>(unwrap(type));
  return wrap(_type.getReceiver());
}

intptr_t mlirGoFunctionTypeGetNumInputs(MlirType type)
{
  const auto _type = mlir::go::cast<mlir::go::FunctionType>(unwrap(type));
  return _type.getNumInputs();
}

MlirType mlirGoFunctionTypeGetInput(MlirType type, intptr_t index)
{
  const auto _type = mlir::go::cast<mlir::go::FunctionType>(unwrap(type));
  return wrap(_type.getInput(index));
}

intptr_t mlirGoFunctionTypeGetNumResults(MlirType type)
{
  const auto _type = mlir::go::cast<mlir::go::FunctionType>(unwrap(type));
  return _type.getNumResults();
}

MlirType mlirGoFunctionTypeGetResult(MlirType type, intptr_t index)
{
  const auto _type = mlir::go::cast<mlir::go::FunctionType>(unwrap(type));
  return wrap(_type.getResult(index));
}
