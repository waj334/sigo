#include "Go-c/mlir/Operations.h"

#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Support.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Operation.h>

#include "Go/IR/GoOps.h"
#include "Go/Util.h"

using namespace mlir;

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
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _location = unwrap(location);
  const auto _asmStr = mlir::cast<mlir::StringAttr>(unwrap(asmStr));

  ::llvm::SmallVector<::mlir::Attribute> _constraints;
  (void)unwrapList(nConstraints, constraints, _constraints);

  ::llvm::SmallVector<::mlir::Attribute> _registerClobbers;
  (void)unwrapList(nRegisterClobbers, registerClobbers, _registerClobbers);

  ::llvm::SmallVector<::mlir::Value> _operands;
  (void)unwrapList(nOperands, operands, _operands);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::InlineAsmOp::create(
    builder,
    _location,
    _asmStr,
    ::mlir::ArrayAttr::get(_context, _constraints),
    ::mlir::ArrayAttr::get(_context, _registerClobbers),
    _operands);
  return wrap(op);
}

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateAddCOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirValue x,
  const MlirValue y,
  const MlirLocation location)
{
  return ::mlir::go::_createBinOp<::mlir::go::AddCOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateAddFOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirValue x,
  const MlirValue y,
  const MlirLocation location)
{
  return ::mlir::go::_createBinOp<::mlir::go::AddFOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateAddIOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirValue x,
  const MlirValue y,
  const MlirLocation location)
{
  return ::mlir::go::_createBinOp<::mlir::go::AddIOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateAddStrOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirValue x,
  const MlirValue y,
  const MlirLocation location)
{
  return ::mlir::go::_createBinOp<::mlir::go::AddStrOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateAndOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirValue x,
  const MlirValue y,
  const MlirLocation location)
{
  return ::mlir::go::_createBinOp<::mlir::go::AndOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateAndNotOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirValue x,
  const MlirValue y,
  const MlirLocation location)
{
  return ::mlir::go::_createBinOp<::mlir::go::AndNotOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateCmpCOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirAttribute predicate,
  const MlirValue x,
  const MlirValue y,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _predicate = mlir::cast<::mlir::go::CmpFPredicateAttr>(unwrap(predicate));
  const auto _x = unwrap(x);
  const auto _y = unwrap(y);
  const auto _resultType = unwrap(resultType);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op =
    ::mlir::go::CmpCOp::create(builder, _location, _resultType, _predicate.getValue(), _x, _y);
  return wrap(op);
}

MlirOperation mlirGoCreateCmpFOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirAttribute predicate,
  const MlirValue x,
  const MlirValue y,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _predicate = mlir::cast<::mlir::go::CmpFPredicateAttr>(unwrap(predicate));
  const auto _x = unwrap(x);
  const auto _y = unwrap(y);
  const auto _resultType = unwrap(resultType);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op =
    ::mlir::go::CmpFOp::create(builder, _location, _resultType, _predicate.getValue(), _x, _y);
  return wrap(op);
}

MlirOperation mlirGoCreateCmpIOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirAttribute predicate,
  const MlirValue x,
  const MlirValue y,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _predicate = mlir::cast<::mlir::go::CmpIPredicateAttr>(unwrap(predicate));
  const auto _x = unwrap(x);
  const auto _y = unwrap(y);
  const auto _resultType = unwrap(resultType);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op =
    ::mlir::go::CmpIOp::create(builder, _location, _resultType, _predicate.getValue(), _x, _y);
  return wrap(op);
}

MlirOperation mlirGoCreateCmpInterfaceOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirValue x,
  const MlirValue y,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _x = unwrap(x);
  const auto _y = unwrap(y);
  const auto _resultType = unwrap(resultType);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::CmpInterfaceOp::create(builder, _location, _resultType, _x, _y);
  return wrap(op);
}

MlirOperation mlirGoCreateCmpStringOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirAttribute predicate,
  const MlirValue x,
  const MlirValue y,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _predicate = mlir::cast<::mlir::go::CmpPredicateAttr>(unwrap(predicate));
  const auto _x = unwrap(x);
  const auto _y = unwrap(y);
  const auto _resultType = unwrap(resultType);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op =
    ::mlir::go::CmpStringOp::create(builder, _location, _resultType, _predicate, _x, _y);
  return wrap(op);
}

MlirOperation mlirGoCreateCmpNilOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirValue x,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _x = unwrap(x);
  const auto _resultType = unwrap(resultType);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::CmpNilOp::create(builder, _location, _resultType, _x);
  return wrap(op);
}

MlirOperation mlirGoCreateDivCOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirValue x,
  const MlirValue y,
  const MlirLocation location)
{
  return ::mlir::go::_createBinOp<::mlir::go::DivCOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateDivFOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirValue x,
  const MlirValue y,
  const MlirLocation location)
{
  return ::mlir::go::_createBinOp<::mlir::go::DivFOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateDivSIOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirValue x,
  const MlirValue y,
  const MlirLocation location)
{
  return ::mlir::go::_createBinOp<::mlir::go::DivSIOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateDivUIOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirValue x,
  const MlirValue y,
  const MlirLocation location)
{
  return ::mlir::go::_createBinOp<::mlir::go::DivUIOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateMulCOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirValue x,
  const MlirValue y,
  const MlirLocation location)
{
  return ::mlir::go::_createBinOp<::mlir::go::MulCOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateMulFOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirValue x,
  const MlirValue y,
  const MlirLocation location)
{
  return ::mlir::go::_createBinOp<::mlir::go::MulFOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateMulIOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirValue x,
  const MlirValue y,
  const MlirLocation location)
{
  return ::mlir::go::_createBinOp<::mlir::go::MulIOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateOrOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirValue x,
  const MlirValue y,
  const MlirLocation location)
{
  return ::mlir::go::_createBinOp<::mlir::go::OrOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateRemFOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirValue x,
  const MlirValue y,
  const MlirLocation location)
{
  return ::mlir::go::_createBinOp<::mlir::go::RemFOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateRemSIOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirValue x,
  const MlirValue y,
  const MlirLocation location)
{
  return ::mlir::go::_createBinOp<::mlir::go::RemSIOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateRemUIOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirValue x,
  const MlirValue y,
  const MlirLocation location)
{
  return ::mlir::go::_createBinOp<::mlir::go::RemUIOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateShlOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirValue x,
  const MlirValue y,
  const MlirLocation location)
{
  return ::mlir::go::_createBinOp<::mlir::go::ShlOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateShrUIOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirValue x,
  const MlirValue y,
  const MlirLocation location)
{
  return ::mlir::go::_createBinOp<::mlir::go::ShrUIOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateShrSIOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirValue x,
  const MlirValue y,
  const MlirLocation location)
{
  return ::mlir::go::_createBinOp<::mlir::go::ShrSIOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateSubCOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirValue x,
  const MlirValue y,
  const MlirLocation location)
{
  return ::mlir::go::_createBinOp<::mlir::go::SubCOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateSubFOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirValue x,
  const MlirValue y,
  const MlirLocation location)
{
  return ::mlir::go::_createBinOp<::mlir::go::SubFOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateSubIOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirValue x,
  const MlirValue y,
  const MlirLocation location)
{
  return ::mlir::go::_createBinOp<::mlir::go::SubIOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateXorOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirValue x,
  const MlirValue y,
  const MlirLocation location)
{
  return ::mlir::go::_createBinOp<::mlir::go::XorOp>(context, resultType, x, y, location);
}

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateComplementOperation(
  const MlirContext context,
  const MlirValue x,
  const MlirLocation location)
{
  return ::mlir::go::_createUnOp<::mlir::go::ComplementOp>(context, x, location);
}

MlirOperation
mlirGoCreateNegCOperation(const MlirContext context, const MlirValue x, const MlirLocation location)
{
  return ::mlir::go::_createUnOp<::mlir::go::NegCOp>(context, x, location);
}

MlirOperation
mlirGoCreateNegFOperation(const MlirContext context, const MlirValue x, const MlirLocation location)
{
  return ::mlir::go::_createUnOp<::mlir::go::NegFOp>(context, x, location);
}

MlirOperation
mlirGoCreateNegIOperation(const MlirContext context, const MlirValue x, const MlirLocation location)
{
  return ::mlir::go::_createUnOp<::mlir::go::NegIOp>(context, x, location);
}

MlirOperation
mlirGoCreateNotOperation(const MlirContext context, const MlirValue x, const MlirLocation location)
{
  return ::mlir::go::_createUnOp<::mlir::go::NotOp>(context, x, location);
}

//===----------------------------------------------------------------------===//
// Map Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateMapUpdateOperation(
  const MlirContext context,
  const MlirValue map,
  const MlirValue key,
  const MlirValue value,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _map = unwrap(map);
  const auto _key = unwrap(key);
  const auto _value = unwrap(value);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::MapUpdateOp::create(builder, _location, _map, _key, _value);
  return wrap(op);
}

MlirOperation mlirGoCreateMapLookupOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirValue map,
  const MlirValue key,
  const bool hasOk,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _resultType = unwrap(resultType);
  const auto _map = unwrap(map);
  const auto _key = unwrap(key);
  const auto _location = unwrap(location);

  const auto boolType = hasOk ? mlir::go::BooleanType::get(_context) : mlir::Type();

  mlir::OpBuilder builder(_context);
  mlir::Operation* op =
    ::mlir::go::MapLookupOp::create(builder, _location, _resultType, boolType, _map, _key);
  return wrap(op);
}

MlirOperation mlirGoCreateMapRangeOp(
  const MlirContext context,
  const MlirValue value,
  const MlirBlock bodyDest,
  const MlirBlock exitDest,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _value = unwrap(value);
  const auto _bodyDest = unwrap(bodyDest);
  const auto _exitDest = unwrap(exitDest);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op =
    ::mlir::go::MapRangeOp::create(builder, _location, _value, _bodyDest, _exitDest);
  return wrap(op);
}

//===----------------------------------------------------------------------===//
// Memory Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateAllocaOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirType elementType,
  const int numElements,
  const bool isHeap,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _resultType = unwrap(resultType);
  const auto _elementType = unwrap(elementType);
  const auto _location = unwrap(location);
  int _numElements = 1;
  if (numElements > 0)
  {
    _numElements = numElements;
  }
  const auto _heap = isHeap ? UnitAttr::get(_context) : UnitAttr();

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::AllocaOp::create(
    builder, _location, _resultType, _elementType, _numElements, _heap, StringAttr());
  return wrap(op);
}

void mlirGoAllocaOperationSetName(const MlirOperation op, const MlirStringRef name)
{
  auto _op = mlir::cast<::mlir::go::AllocaOp>(unwrap(op));
  const auto _name = unwrap(name);
  _op.setVarName(_name);
}

void mlirGoAllocaOperationSetIsHeap(const MlirOperation op, const bool isHeap)
{
  auto _op = mlir::cast<::mlir::go::AllocaOp>(unwrap(op));
  _op.setHeap(isHeap);
}

MlirOperation mlirGoCreateLoadOperation(
  const MlirContext context,
  const MlirValue x,
  const MlirType resultType,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _x = unwrap(x);
  const auto _resultType = unwrap(resultType);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::LoadOp::create(builder, _location, _resultType, _x);
  return wrap(op);
}

MlirOperation mlirGoCreateVolatileLoadOperation(
  const MlirContext context,
  const MlirValue x,
  const MlirType resultType,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _x = unwrap(x);
  const auto _resultType = unwrap(resultType);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::LoadOp::create(
    builder, _location, _resultType, _x, mlir::UnitAttr::get(_context), mlir::UnitAttr());
  return wrap(op);
}

MlirOperation mlirGoCreateAtomicLoadOperation(
  const MlirContext context,
  const MlirValue x,
  const MlirType resultType,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _x = unwrap(x);
  const auto _resultType = unwrap(resultType);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::LoadOp::create(
    builder, _location, _resultType, _x, mlir::UnitAttr(), mlir::UnitAttr::get(_context));
  return wrap(op);
}

MlirOperation mlirGoCreateStoreOperation(
  const MlirContext context,
  const MlirValue value,
  const MlirValue address,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _value = unwrap(value);
  const auto _address = unwrap(address);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::StoreOp::create(
    builder, _location, _value, _address, mlir::UnitAttr(), mlir::UnitAttr());
  return wrap(op);
}

MlirOperation mlirGoCreateVolatileStoreOperation(
  const MlirContext context,
  const MlirValue value,
  const MlirValue address,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _value = unwrap(value);
  const auto _address = unwrap(address);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::StoreOp::create(
    builder, _location, _value, _address, mlir::UnitAttr::get(_context), mlir::UnitAttr());
  return wrap(op);
}

MlirOperation mlirGoCreateAtomicStoreOperation(
  const MlirContext context,
  const MlirValue value,
  const MlirValue address,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _value = unwrap(value);
  const auto _address = unwrap(address);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::StoreOp::create(
    builder, _location, _value, _address, mlir::UnitAttr(), mlir::UnitAttr::get(_context));
  return wrap(op);
}

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
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _addr = unwrap(addr);
  const auto _baseType = unwrap(baseType);
  const auto _type = unwrap(type);
  const auto _location = unwrap(location);

  ::llvm::SmallVector<::mlir::Value> _dynamicIndices;
  (void)unwrapList(nDynamicIndices, dynamicIndices, _dynamicIndices);

  const ::llvm::ArrayRef<int32_t> _constIndices(constIndices, nConstIndices);
  const ::llvm::ArrayRef<bool> _indexFlags(indexFlags, nIndexFlags);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::GetElementPointerOp::create(
    builder, _location, _type, _addr, _baseType, _dynamicIndices, _constIndices, _indexFlags);
  return wrap(op);
}

MlirOperation mlirGoCreateGlobalOperation(
  const MlirContext context,
  const MlirAttribute linkage,
  const MlirStringRef symbol,
  const MlirStringRef section,
  const MlirType type,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _type = unwrap(type);
  const auto _location = unwrap(location);
  const auto _symbol = unwrap(symbol);
  const auto _section = unwrap(section);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::GlobalOp::create(
    builder,
    _location,
    _type,
    mlir::StringAttr::get(_context, _symbol),
    _section.size() == 0 ? mlir::StringAttr() : mlir::StringAttr::get(_context, _section));

  if (!mlirAttributeIsNull(linkage))
  {
    const auto _linkage = unwrap(linkage);
    op->setAttr("llvm.linkage", _linkage);
  }

  return wrap(op);
}

MlirOperation mlirGoCreateYieldOperation(
  const MlirContext context,
  const MlirValue value,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _value = unwrap(value);
  const auto _location = unwrap(location);
  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::YieldOp::create(builder, _location, _value);
  return wrap(op);
}

MlirOperation mlirGoCreateSliceOperation(
  const MlirContext context,
  const MlirValue input,
  const MlirValue low,
  const MlirValue high,
  const MlirValue max,
  const MlirType resultType,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _input = unwrap(input);
  const auto _low = !mlirValueIsNull(low) ? unwrap(low) : mlir::Value();
  const auto _high = !mlirValueIsNull(high) ? unwrap(high) : mlir::Value();
  const auto _max = !mlirValueIsNull(max) ? unwrap(max) : mlir::Value();
  const auto _resultType = unwrap(resultType);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op =
    ::mlir::go::SliceOp::create(builder, _location, _resultType, _input, _low, _high, _max);

  const auto operandSegmentSizesAttr =
    DenseI32ArrayAttr::get(_context, { _low ? 1 : 0, _high ? 1 : 0, _max ? 1 : 0 });
  op->setAttr("operandSegmentSizes", operandSegmentSizesAttr);

  return wrap(op);
}

MlirOperation mlirGoCreateAddressOfOperation(
  const MlirContext context,
  const MlirStringRef symbol,
  const MlirType resultType,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _resultType = unwrap(resultType);
  const auto _location = unwrap(location);
  const auto _symbol = unwrap(symbol);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::AddressOfOp::create(
    builder, _location, _resultType, mlir::FlatSymbolRefAttr::get(_context, _symbol));
  return wrap(op);
}

MlirOperation mlirGoCreateNilPointerCheckOperation(
  const MlirContext context,
  const MlirValue addr,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _addr = unwrap(addr);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  return wrap(mlir::go::NilPointerCheckOp::create(builder, _location, _addr));
}

//===----------------------------------------------------------------------===//
// Slice Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateSliceAddrOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirValue slice,
  const MlirValue index,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _resultType = unwrap(resultType);
  const auto _slice = unwrap(slice);
  const auto _index = unwrap(index);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op =
    ::mlir::go::SliceAddrOp::create(builder, _location, _resultType, _slice, _index);
  return wrap(op);
}

//===----------------------------------------------------------------------===//
// String Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateStringAddrOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirValue slice,
  const MlirValue index,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _resultType = unwrap(resultType);
  const auto _slice = unwrap(slice);
  const auto _index = unwrap(index);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op =
    ::mlir::go::StringAddrOp::create(builder, _location, _resultType, _slice, _index);
  return wrap(op);
}

MlirOperation mlirGoCreateStringRangeOp(
  const MlirContext context,
  const MlirValue value,
  const MlirBlock bodyDest,
  const MlirBlock exitDest,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _value = unwrap(value);
  const auto _bodyDest = unwrap(bodyDest);
  const auto _exitDest = unwrap(exitDest);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op =
    ::mlir::go::StringRangeOp::create(builder, _location, _value, _bodyDest, _exitDest);
  return wrap(op);
}

//===----------------------------------------------------------------------===//
// Aggregate Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateExtractOperation(
  const MlirContext context,
  const uint64_t index,
  const MlirType fieldType,
  const MlirValue structValue,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _fieldType = unwrap(fieldType);
  const auto _structValue = unwrap(structValue);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op =
    ::mlir::go::ExtractOp::create(builder, _location, _fieldType, index, _structValue);
  return wrap(op);
}

MlirOperation mlirGoCreateInsertOperation(
  const MlirContext context,
  const uint64_t index,
  const MlirValue value,
  const MlirValue structValue,
  const MlirType structType,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _value = unwrap(value);
  const auto _structType = unwrap(structType);
  const auto _structValue = unwrap(structValue);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op =
    ::mlir::go::InsertOp::create(builder, _location, _structType, _value, index, _structValue);
  return wrap(op);
}

//===----------------------------------------------------------------------===//
// Constant Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateConstantOperation(
  const MlirContext context,
  const MlirAttribute value,
  const MlirAttribute symbol,
  const MlirType type,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _type = unwrap(type);
  const auto _location = unwrap(location);
  const auto _value = unwrap(value);

  mlir::StringAttr _symbol;
  if (!mlirAttributeIsNull(symbol))
  {
    _symbol = mlir::cast<mlir::StringAttr>(unwrap(symbol));
  }

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::ConstantOp::create(builder, _location, _type, _value, _symbol);
  return wrap(op);
}

MlirOperation mlirGoCreateGlobalConstantOperation(
  const MlirContext context,
  const MlirAttribute value,
  const MlirAttribute symbol,
  const MlirLocation location)
{

  const auto _context = unwrap(context);
  const auto _location = unwrap(location);
  const auto _symbol = mlir::cast<mlir::StringAttr>(unwrap(symbol));

  mlir::Attribute _value;
  if (!mlirAttributeIsNull(value))
  { // nullable handle
    _value = unwrap(value);
  }

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::GlobalConstantOp::create(builder, _location, _symbol, _value);
  return wrap(op);
}

void mlirGoGlobalConstantOperationAddBody(const MlirOperation op, const MlirBlock body)
{
  auto _op = mlir::cast<mlir::go::GlobalConstantOp>(unwrap(op));
  const auto _body = unwrap(body);
  _op.getBody().push_back(_body);
}

//===----------------------------------------------------------------------===//
// Casting Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateBitcastOperation(
  const MlirContext context,
  const MlirValue value,
  const MlirType type,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _value = unwrap(value);
  const auto _type = unwrap(type);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::BitcastOp::create(builder, _location, _type, _value);
  return wrap(op);
}

MlirOperation mlirGoCreateComplexExtendOperation(
  const MlirContext context,
  const MlirValue value,
  const MlirType type,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _value = unwrap(value);
  const auto _type = unwrap(type);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::ComplexExtendOp::create(builder, _location, _type, _value);
  return wrap(op);
}

MlirOperation mlirGoCreateComplexTruncateOperation(
  const MlirContext context,
  const MlirValue value,
  const MlirType type,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _value = unwrap(value);
  const auto _type = unwrap(type);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::ComplexTruncateOp::create(builder, _location, _type, _value);
  return wrap(op);
}

MlirOperation mlirGoCreateIntToPtrOperation(
  const MlirContext context,
  const MlirValue value,
  const MlirType type,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _value = unwrap(value);
  const auto _type = unwrap(type);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::IntToPtrOp::create(builder, _location, _type, _value);
  return wrap(op);
}

MlirOperation mlirGoCreatePtrToIntOperation(
  const MlirContext context,
  const MlirValue value,
  const MlirType type,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _value = unwrap(value);
  const auto _type = unwrap(type);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::PtrToIntOp::create(builder, _location, _type, _value);
  return wrap(op);
}

MlirOperation mlirGoCreateFloatTruncateOperation(
  const MlirContext context,
  const MlirValue value,
  const MlirType type,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _value = unwrap(value);
  const auto _type = unwrap(type);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::FloatTruncateOp::create(builder, _location, _type, _value);
  return wrap(op);
}

MlirOperation mlirGoCreateIntTruncateOperation(
  const MlirContext context,
  const MlirValue value,
  const MlirType type,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _value = unwrap(value);
  const auto _type = unwrap(type);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::IntTruncateOp::create(builder, _location, _type, _value);
  return wrap(op);
}

MlirOperation mlirGoCreateFloatExtendOperation(
  const MlirContext context,
  const MlirValue value,
  const MlirType type,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _value = unwrap(value);
  const auto _type = unwrap(type);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::FloatExtendOp::create(builder, _location, _type, _value);
  return wrap(op);
}

MlirOperation mlirGoCreateSignedExtendOperation(
  const MlirContext context,
  const MlirValue value,
  const MlirType type,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _value = unwrap(value);
  const auto _type = unwrap(type);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::SignedExtendOp::create(builder, _location, _type, _value);
  return wrap(op);
}

MlirOperation mlirGoCreateZeroExtendOperation(
  const MlirContext context,
  const MlirValue value,
  const MlirType type,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _value = unwrap(value);
  const auto _type = unwrap(type);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::ZeroExtendOp::create(builder, _location, _type, _value);
  return wrap(op);
}

MlirOperation mlirGoCreateFloatToUnsignedIntOperation(
  const MlirContext context,
  const MlirValue value,
  const MlirType type,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _value = unwrap(value);
  const auto _type = unwrap(type);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::FloatToUnsignedIntOp::create(builder, _location, _type, _value);
  return wrap(op);
}

MlirOperation mlirGoCreateFloatToSignedIntOperation(
  const MlirContext context,
  const MlirValue value,
  const MlirType type,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _value = unwrap(value);
  const auto _type = unwrap(type);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::FloatToSignedIntOp::create(builder, _location, _type, _value);
  return wrap(op);
}

MlirOperation mlirGoCreateUnsignedIntToFloatOperation(
  const MlirContext context,
  const MlirValue value,
  const MlirType type,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _value = unwrap(value);
  const auto _type = unwrap(type);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::UnsignedIntToFloatOp::create(builder, _location, _type, _value);
  return wrap(op);
}

MlirOperation mlirGoCreateSignedIntToFloatOperation(
  const MlirContext context,
  const MlirValue value,
  const MlirType type,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _value = unwrap(value);
  const auto _type = unwrap(type);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::SignedIntToFloatOp::create(builder, _location, _type, _value);
  return wrap(op);
}

MlirOperation mlirGoCreateFunctionToPointerOperation(
  const MlirContext context,
  const MlirValue value,
  const MlirType type,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _value = unwrap(value);
  const auto _type = unwrap(type);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::FunctionToPointerOp::create(builder, _location, _type, _value);
  return wrap(op);
}

MlirOperation mlirGoCreatePointerToFunctionOperation(
  const MlirContext context,
  const MlirValue value,
  const MlirType type,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _value = unwrap(value);
  const auto _type = unwrap(type);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::PointerToFunctionOp::create(builder, _location, _type, _value);
  return wrap(op);
}

MlirOperation mlirGoCreateChangeInterfaceOperation(
  const MlirContext context,
  const MlirValue value,
  const MlirType type,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _value = unwrap(value);
  const auto _type = unwrap(type);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::ChangeInterfaceOp::create(builder, _location, _type, _value);
  return wrap(op);
}

MlirOperation mlirGoCreateTypeAssertOperation(
  const MlirContext context,
  const MlirValue value,
  const int nResults,
  const MlirType* results,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _value = unwrap(value);

  SmallVector<mlir::Type> _results;
  (void)unwrapList(nResults, results, _results);

  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::TypeAssertOp::create(builder, _location, _results, _value);
  return wrap(op);
}

MlirOperation mlirGoCreateStringToSliceOperation(
  const MlirContext context,
  const MlirValue value,
  const MlirType type,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _value = unwrap(value);
  const auto _type = unwrap(type);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::StringToSliceOp::create(builder, _location, _type, _value);
  return wrap(op);
}

MlirOperation mlirGoCreateSliceToStringOperation(
  const MlirContext context,
  const MlirValue value,
  const MlirType type,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _value = unwrap(value);
  const auto _type = unwrap(type);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::SliceToStringOp::create(builder, _location, _type, _value);
  return wrap(op);
}

//===----------------------------------------------------------------------===//
// Function Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoGetFunction(
  const MlirContext context,
  const MlirStringRef symbol,
  const MlirType type,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _symbol = unwrap(symbol);
  const auto _type = unwrap(type);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::func::ConstantOp::create(
    builder, _location, _type, ::mlir::SymbolRefAttr::get(_context, _symbol));
  return wrap(op);
}

//===----------------------------------------------------------------------===//
// Builtin Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreatePanicOperation(
  const MlirContext context,
  const MlirValue value,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _location = unwrap(location);
  const auto _value = unwrap(value);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::PanicOp::create(builder, _location, _value);
  return wrap(op);
}

MlirOperation mlirGoCreateRecoverOperation(
  const MlirContext context,
  const MlirType type,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _type = unwrap(type);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::RecoverOp::create(builder, _location, _type);
  return wrap(op);
}

//===----------------------------------------------------------------------===//
// Atomic Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateAtomicAddIOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirValue addr,
  const MlirValue delta,
  const MlirLocation location)
{
  return ::mlir::go::_createBinOp<::mlir::go::AtomicAddIOp>(
    context, resultType, addr, delta, location);
}

MlirOperation mlirGoCreateAtomicCompareAndSwapOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirValue addr,
  const MlirValue old,
  const MlirValue value,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _resultType = unwrap(resultType);
  const auto _addr = unwrap(addr);
  const auto _old = unwrap(old);
  const auto _value = unwrap(value);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::AtomicCompareAndSwapIOp::create(
    builder, _location, _resultType, _addr, _old, _value);
  return wrap(op);
}

MlirOperation mlirGoCreateAtomicSwapOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirValue addr,
  const MlirValue value,
  const MlirLocation location)
{
  return ::mlir::go::_createBinOp<::mlir::go::AtomicSwapIOp>(
    context, resultType, addr, value, location);
}

//===----------------------------------------------------------------------===//
// Control Flow Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateBranchOperation(
  const MlirContext context,
  const MlirBlock dest,
  const int nDestOperands,
  const MlirValue* destOperands,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _dest = unwrap(dest);
  const auto _location = unwrap(location);

  ::llvm::SmallVector<::mlir::Value> _destOperands;
  (void)unwrapList(nDestOperands, destOperands, _destOperands);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::BranchOp::create(builder, _location, _destOperands, _dest);
  return wrap(op);
}

MlirOperation mlirGoCreateCondBranchOperation(
  const MlirContext context,
  const MlirValue condition,
  const MlirBlock trueDest,
  const int nTrueDestOperands,
  const MlirValue* trueDestOperands,
  const MlirBlock falseDest,
  const int nFalseDestOperands,
  const MlirValue* falseDestOperands,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _condition = unwrap(condition);
  const auto _trueDest = unwrap(trueDest);
  const auto _falseDest = unwrap(falseDest);
  const auto _location = unwrap(location);

  ::llvm::SmallVector<::mlir::Value> _trueDestOperands;
  (void)unwrapList(nTrueDestOperands, trueDestOperands, _trueDestOperands);

  ::llvm::SmallVector<::mlir::Value> _falseDestOperands;
  (void)unwrapList(nFalseDestOperands, falseDestOperands, _falseDestOperands);

  const auto operandSegmentSizesAttr = DenseI32ArrayAttr::get(
    _context, { 1, static_cast<int>(nTrueDestOperands), static_cast<int>(nFalseDestOperands) });

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::CondBranchOp::create(
    builder, _location, _condition, _trueDestOperands, _falseDestOperands, _trueDest, _falseDest);
  op->setAttr("operandSegmentSizes", operandSegmentSizesAttr);
  return wrap(op);
}

MlirOperation mlirGoCreateReturnOperation(
  const MlirContext context,
  const int nOperands,
  const MlirValue* operands,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _location = unwrap(location);

  ::llvm::SmallVector<::mlir::Value> _operands;
  (void)unwrapList(nOperands, operands, _operands);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::ReturnOp::create(builder, _location, _operands);
  return wrap(op);
}

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
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _callee = unwrap(callee);
  const auto _location = unwrap(location);

  ::llvm::SmallVector<::mlir::Type> _resultTypes;
  (void)unwrapList(nResultTypes, resultTypes, _resultTypes);

  ::llvm::SmallVector<::mlir::Value> _operands;
  (void)unwrapList(nOperands, operands, _operands);

  mlir::OpBuilder builder(_context);
  auto op = ::mlir::go::CallOp::create(builder, _location, _resultTypes, _callee, _operands);

  return wrap(op);
}

MlirOperation mlirGoCreateClosureCallOperation(
  const MlirContext context,
  const MlirAttribute signature,
  const MlirValue callee,
  const int nResultTypes,
  const MlirType* resultTypes,
  const int nOperands,
  const MlirValue* operands,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _callee = unwrap(callee);
  const auto _signature = mlir::cast<mlir::TypeAttr>(unwrap(signature));
  const auto _location = unwrap(location);

  ::llvm::SmallVector<::mlir::Type> _resultTypes;
  (void)unwrapList(nResultTypes, resultTypes, _resultTypes);

  ::llvm::SmallVector<::mlir::Value> _operands;
  (void)unwrapList(nOperands, operands, _operands);

  mlir::OpBuilder builder(_context);
  auto op =
    ::mlir::go::CallOp::create(builder, _location, _signature, _resultTypes, _callee, _operands);
  return wrap(op);
}

MlirOperation mlirGoCreateCallIndirectOperation(
  const MlirContext context,
  const MlirValue callee,
  const int nResultTypes,
  const MlirType* resultTypes,
  const int nOperands,
  const MlirValue* operands,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _callee = unwrap(callee);
  const auto _location = unwrap(location);

  ::llvm::SmallVector<::mlir::Type> _resultTypes;
  (void)unwrapList(nResultTypes, resultTypes, _resultTypes);

  ::llvm::SmallVector<::mlir::Value> _operands;
  (void)unwrapList(nOperands, operands, _operands);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op =
    ::mlir::go::CallIndirectOp::create(builder, _location, _resultTypes, _callee, _operands);
  return wrap(op);
}

MlirOperation mlirGoCreateDeferOperation1(
  const MlirContext context,
  const MlirStringRef sym_name,
  const int nArgs,
  const MlirValue* args,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _sym_name = unwrap(sym_name);
  const auto _location = unwrap(location);

  ::llvm::SmallVector<::mlir::Value> _args;
  (void)unwrapList(nArgs, args, _args);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::DeferOp::create(builder, _location, _sym_name, _args);
  return wrap(op);
}

MlirOperation mlirGoCreateDeferOperation2(
  const MlirContext context,
  const MlirAttribute sym_name,
  const int nArgs,
  const MlirValue* args,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _sym_name = unwrap(sym_name);
  const auto _location = unwrap(location);

  ::llvm::SmallVector<::mlir::Value> _args;
  (void)unwrapList(nArgs, args, _args);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op =
    mlir::TypeSwitch<mlir::Attribute, mlir::Operation*>(_sym_name)
      .Case([&](const mlir::StringAttr attr)
            { return ::mlir::go::DeferOp::create(builder, _location, attr, _args); })
      .Case([&](const mlir::SymbolRefAttr attr)
            { return ::mlir::go::DeferOp::create(builder, _location, attr, _args); })
      .Default(
        [&](mlir::Attribute)
        {
          assert(false && "invalid attribute type");
          return nullptr;
        });
  return wrap(op);
}

MlirOperation mlirGoCreateDeferOperation3(
  const MlirContext context,
  const MlirAttribute signature,
  const MlirValue callee_value,
  const int nArgs,
  const MlirValue* args,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _signature = mlir::cast<mlir::TypeAttr>(unwrap(signature));
  const auto _callee_value = unwrap(callee_value);
  const auto _location = unwrap(location);

  ::llvm::SmallVector<::mlir::Value> _args;
  (void)unwrapList(nArgs, args, _args);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op =
    ::mlir::go::DeferOp::create(builder, _location, _signature, _callee_value, _args);
  return wrap(op);
}

MlirOperation mlirGoCreateDeferOperation4(
  const MlirContext context,
  const MlirValue iface_value,
  const MlirStringRef method_name,
  const int nArgs,
  const MlirValue* args,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _iface_value = unwrap(iface_value);
  const auto _method_name = unwrap(method_name);
  const auto _location = unwrap(location);

  ::llvm::SmallVector<::mlir::Value> _args;
  (void)unwrapList(nArgs, args, _args);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op =
    ::mlir::go::DeferOp::create(builder, _location, _iface_value, _method_name, _args);
  return wrap(op);
}

MlirOperation mlirGoCreateDeferOperation5(
  const MlirContext context,
  const MlirValue iface_value,
  const MlirAttribute method_name,
  const int nArgs,
  const MlirValue* args,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _iface_value = unwrap(iface_value);
  const auto _method_name = mlir::cast<mlir::StringAttr>(unwrap(method_name));
  const auto _location = unwrap(location);

  ::llvm::SmallVector<::mlir::Value> _args;
  (void)unwrapList(nArgs, args, _args);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op =
    ::mlir::go::DeferOp::create(builder, _location, _iface_value, _method_name, _args);
  return wrap(op);
}

MlirOperation mlirGoCreateGoOperation1(
  const MlirContext context,
  const MlirStringRef sym_name,
  const int nArgs,
  const MlirValue* args,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _sym_name = unwrap(sym_name);
  const auto _location = unwrap(location);

  ::llvm::SmallVector<::mlir::Value> _args;
  (void)unwrapList(nArgs, args, _args);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::GoOp::create(builder, _location, _sym_name, _args);
  return wrap(op);
}

MlirOperation mlirGoCreateGoOperation2(
  const MlirContext context,
  const MlirAttribute sym_name,
  const int nArgs,
  const MlirValue* args,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _sym_name = unwrap(sym_name);
  const auto _location = unwrap(location);

  ::llvm::SmallVector<::mlir::Value> _args;
  (void)unwrapList(nArgs, args, _args);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op =
    mlir::TypeSwitch<mlir::Attribute, mlir::Operation*>(_sym_name)
      .Case([&](mlir::StringAttr attr)
            { return ::mlir::go::GoOp::create(builder, _location, attr, _args); })
      .Case([&](mlir::SymbolRefAttr attr)
            { return ::mlir::go::GoOp::create(builder, _location, attr, _args); })
      .Default(
        [&](mlir::Attribute)
        {
          assert(false && "invalid attribute type");
          return nullptr;
        });
  return wrap(op);
}

MlirOperation mlirGoCreateGoOperation3(
  const MlirContext context,
  const MlirAttribute signature,
  const MlirValue callee_value,
  const int nArgs,
  const MlirValue* args,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _signature = mlir::cast<mlir::TypeAttr>(unwrap(signature));
  const auto _callee_value = unwrap(callee_value);
  const auto _location = unwrap(location);

  ::llvm::SmallVector<::mlir::Value> _args;
  (void)unwrapList(nArgs, args, _args);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op =
    ::mlir::go::GoOp::create(builder, _location, _signature, _callee_value, _args);
  return wrap(op);
}

MlirOperation mlirGoCreateGoOperation4(
  const MlirContext context,
  const MlirValue iface_value,
  const MlirStringRef method_name,
  const int nArgs,
  const MlirValue* args,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _iface_value = unwrap(iface_value);
  const auto _method_name = unwrap(method_name);
  const auto _location = unwrap(location);

  ::llvm::SmallVector<::mlir::Value> _args;
  (void)unwrapList(nArgs, args, _args);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op =
    ::mlir::go::GoOp::create(builder, _location, _iface_value, _method_name, _args);
  return wrap(op);
}

MlirOperation mlirGoCreateGoOperation5(
  const MlirContext context,
  const MlirValue iface_value,
  const MlirAttribute method_name,
  const int nArgs,
  const MlirValue* args,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _iface_value = unwrap(iface_value);
  const auto _method_name = mlir::cast<mlir::StringAttr>(unwrap(method_name));
  const auto _location = unwrap(location);

  ::llvm::SmallVector<::mlir::Value> _args;
  (void)unwrapList(nArgs, args, _args);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op =
    ::mlir::go::GoOp::create(builder, _location, _iface_value, _method_name, _args);
  return wrap(op);
}

MlirOperation mlirGoCreateInterfaceCall(
  const MlirContext context,
  const MlirStringRef callee,
  const int nResultTypes,
  const MlirType* resultTypes,
  const MlirValue ifaceValue,
  const int nArgs,
  const MlirValue* args,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _callee = unwrap(callee);
  const auto _ifaceValue = unwrap(ifaceValue);
  const auto _location = unwrap(location);

  ::llvm::SmallVector<::mlir::Type> _resultTypes;
  (void)unwrapList(nResultTypes, resultTypes, _resultTypes);

  ::llvm::SmallVector<::mlir::Value> _args;
  (void)unwrapList(nArgs, args, _args);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::InterfaceCallOp::create(
    builder, _location, _resultTypes, _callee, _ifaceValue, _args);
  return wrap(op);
}

MlirOperation mlirGoCreateBuiltInCallOperation(
  const MlirContext context,
  const MlirStringRef identifier,
  const int nResultTypes,
  const MlirType* resultTypes,
  const int nOperands,
  const MlirValue* operands,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _identifier = unwrap(identifier);
  const auto _location = unwrap(location);

  ::llvm::SmallVector<::mlir::Type> _resultTypes;
  (void)unwrapList(nResultTypes, resultTypes, _resultTypes);

  ::llvm::SmallVector<::mlir::Value> _operands;
  (void)unwrapList(nOperands, operands, _operands);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op =
    ::mlir::go::BuiltInCallOp::create(builder, _location, _resultTypes, _identifier, _operands);
  return wrap(op);
}

//===----------------------------------------------------------------------===//
// Value Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateZeroOperation(
  const MlirContext context,
  const MlirType type,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _type = unwrap(type);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::ZeroOp::create(builder, _location, _type);
  return wrap(op);
}

MlirOperation mlirGoCreateComplexOperation(
  const MlirContext context,
  const MlirType type,
  const MlirValue real,
  const MlirValue imag,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _type = unwrap(type);
  const auto _real = unwrap(real);
  const auto _imag = unwrap(imag);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::ComplexOp::create(builder, _location, _type, _real, _imag);
  return wrap(op);
}

MlirOperation mlirGoCreateImagOperation(
  const MlirContext context,
  const MlirType type,
  const MlirValue value,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _type = unwrap(type);
  const auto _value = unwrap(value);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::ImagOp::create(builder, _location, _type, _value);
  return wrap(op);
}

MlirOperation mlirGoCreateRealOperation(
  const MlirContext context,
  const MlirType type,
  const MlirValue value,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _type = unwrap(type);
  const auto _value = unwrap(value);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::RealOp::create(builder, _location, _type, _value);
  return wrap(op);
}

MlirOperation mlirGoCreateMakeMapOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirValue capacity,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _resultType = unwrap(resultType);
  const auto _location = unwrap(location);

  mlir::Value _capacity;
  if (!mlirValueIsNull(capacity))
  {
    _capacity = unwrap(capacity);
  }

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::MakeMapOp::create(builder, _location, _resultType, _capacity);
  return wrap(op);
}

MlirOperation mlirGoCreateMakeSliceOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirValue length,
  const MlirValue capacity,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _resultType = unwrap(resultType);
  const auto _length = unwrap(length);
  const auto _location = unwrap(location);

  mlir::Value _capacity;
  if (!mlirValueIsNull(capacity))
  {
    _capacity = unwrap(capacity);
  }

  mlir::OpBuilder builder(_context);
  mlir::Operation* op =
    ::mlir::go::MakeSliceOp::create(builder, _location, _resultType, _length, _capacity);
  return wrap(op);
}

MlirOperation mlirGoCreateMakeInterfaceOperation(
  const MlirContext context,
  const MlirType resultType,
  const MlirType type,
  const MlirValue value,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _resultType = unwrap(resultType);
  const auto _type = unwrap(type);
  const auto _value = unwrap(value);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op =
    ::mlir::go::MakeInterfaceOp::create(builder, _location, _resultType, _type, _value);
  return wrap(op);
}

MlirOperation mlirGoCreateChanRecvOp(
  const MlirContext context,
  const int nResultTypes,
  const MlirType* resultTypes,
  const MlirValue channel,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _channel = unwrap(channel);
  const auto _location = unwrap(location);

  mlir::SmallVector<mlir::Type> _resultTypes;
  (void)unwrapList(nResultTypes, resultTypes, _resultTypes);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::ChanRecvOp::create(builder, _location, _resultTypes, _channel);
  return wrap(op);
}

MlirOperation mlirGoCreateChanSendOp(
  const MlirContext context,
  const MlirValue channel,
  const MlirValue value,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _channel = unwrap(channel);
  const auto _value = unwrap(value);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::ChanSendOp::create(builder, _location, _channel, _value);
  return wrap(op);
}

MlirOperation mlirGoCreateChanRangeOp(
  const MlirContext context,
  const MlirValue channel,
  const MlirBlock bodyDest,
  const MlirBlock exitDest,
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _channel = unwrap(channel);
  const auto _bodyDest = unwrap(bodyDest);
  const auto _exitDest = unwrap(exitDest);
  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op =
    ::mlir::go::ChanRangeOp::create(builder, _location, _channel, _bodyDest, _exitDest);
  return wrap(op);
}

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
  const MlirLocation location)
{
  const auto _context = unwrap(context);
  const auto _hasDefault = hasDefault ? mlir::UnitAttr::get(_context) : mlir::UnitAttr();
  const auto _send = mlir::cast<mlir::DenseBoolArrayAttr>(unwrap(send));

  mlir::SmallVector<mlir::Value> _chans;
  (void)unwrapList(nChans, chans, _chans);

  const auto _defaultDest = unwrap(defaultDest);
  const auto _exitDest = unwrap(exitDest);

  mlir::SmallVector<mlir::Block*> _cases;
  (void)unwrapList(nCases, cases, _cases);

  const auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = ::mlir::go::ChanSelectOp::create(
    builder, _location, _hasDefault, _send, _chans, _defaultDest, _exitDest, _cases);
  return wrap(op);
}
