#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Support.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Operation.h>

#include "Go-c/mlir/Operations.h"
#include "Go/IR/GoOps.h"
#include "Go/Util.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateAddCOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                        MlirLocation location) {
    return ::mlir::go::_createBinOp<::mlir::go::AddCOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateAddFOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                        MlirLocation location) {
    return ::mlir::go::_createBinOp<::mlir::go::AddFOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateAddIOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                        MlirLocation location) {
    return ::mlir::go::_createBinOp<::mlir::go::AddIOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateAddStrOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                          MlirLocation location) {
    return ::mlir::go::_createBinOp<::mlir::go::AddStrOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateAndOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                       MlirLocation location) {
    return ::mlir::go::_createBinOp<::mlir::go::AndOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateAndNotOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                          MlirLocation location) {
    return ::mlir::go::_createBinOp<::mlir::go::AndNotOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateCmpCOperation(MlirContext context, MlirType resultType, MlirAttribute predicate, MlirValue x,
                                        MlirValue y, MlirLocation location) {
    auto _context = unwrap(context);
    auto _predicate = unwrap(predicate).dyn_cast<::mlir::go::CmpFPredicateAttr>();
    auto _x = unwrap(x);
    auto _y = unwrap(y);
    auto _resultType = unwrap(resultType);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::CmpCOp>(_location, _resultType, _predicate.getValue(), _x, _y);
    return wrap(op);
}

MlirOperation mlirGoCreateCmpFOperation(MlirContext context, MlirType resultType, MlirAttribute predicate, MlirValue x,
                                        MlirValue y, MlirLocation location) {
    auto _context = unwrap(context);
    auto _predicate = unwrap(predicate).dyn_cast<::mlir::go::CmpFPredicateAttr>();
    auto _x = unwrap(x);
    auto _y = unwrap(y);
    auto _resultType = unwrap(resultType);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::CmpFOp>(_location, _resultType, _predicate.getValue(), _x, _y);
    return wrap(op);
}

MlirOperation mlirGoCreateCmpIOperation(MlirContext context, MlirType resultType, MlirAttribute predicate, MlirValue x,
                                        MlirValue y, MlirLocation location) {
    auto _context = unwrap(context);
    auto _predicate = unwrap(predicate).dyn_cast<::mlir::go::CmpIPredicateAttr>();
    auto _x = unwrap(x);
    auto _y = unwrap(y);
    auto _resultType = unwrap(resultType);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::CmpIOp>(_location, _resultType, _predicate.getValue(), _x, _y);
    return wrap(op);
}

MlirOperation mlirGoCreateDivCOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                        MlirLocation location) {
    return ::mlir::go::_createBinOp<::mlir::go::DivCOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateDivFOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                        MlirLocation location) {
    return ::mlir::go::_createBinOp<::mlir::go::DivFOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateDivSIOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                         MlirLocation location) {
    return ::mlir::go::_createBinOp<::mlir::go::DivSIOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateDivUIOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                         MlirLocation location) {
    return ::mlir::go::_createBinOp<::mlir::go::DivUIOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateMulCOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                        MlirLocation location) {
    return ::mlir::go::_createBinOp<::mlir::go::MulCOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateMulFOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                        MlirLocation location) {
    return ::mlir::go::_createBinOp<::mlir::go::MulFOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateMulIOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                        MlirLocation location) {
    return ::mlir::go::_createBinOp<::mlir::go::MulIOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateOrOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                      MlirLocation location) {
    return ::mlir::go::_createBinOp<::mlir::go::OrOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateRemFOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                        MlirLocation location) {
    return ::mlir::go::_createBinOp<::mlir::go::RemFOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateRemSIOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                         MlirLocation location) {
    return ::mlir::go::_createBinOp<::mlir::go::RemSIOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateRemUIOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                         MlirLocation location) {
    return ::mlir::go::_createBinOp<::mlir::go::RemUIOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateShlOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                       MlirLocation location) {
    return ::mlir::go::_createBinOp<::mlir::go::ShlOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateShrUIOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                         MlirLocation location) {
    return ::mlir::go::_createBinOp<::mlir::go::ShrUIOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateShrSIOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                         MlirLocation location) {
    return ::mlir::go::_createBinOp<::mlir::go::ShrSIOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateSubCOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                        MlirLocation location) {
    return ::mlir::go::_createBinOp<::mlir::go::SubCOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateSubFOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                        MlirLocation location) {
    return ::mlir::go::_createBinOp<::mlir::go::SubFOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateSubIOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                        MlirLocation location) {
    return ::mlir::go::_createBinOp<::mlir::go::SubIOp>(context, resultType, x, y, location);
}

MlirOperation mlirGoCreateXorOperation(MlirContext context, MlirType resultType, MlirValue x, MlirValue y,
                                       MlirLocation location) {
    return ::mlir::go::_createBinOp<::mlir::go::XorOp>(context, resultType, x, y, location);
}

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateComplementOperation(MlirContext context, MlirValue x, MlirLocation location) {
    return ::mlir::go::_createUnOp<::mlir::go::ComplementOp>(context, x, location);
}

MlirOperation mlirGoCreateNegCOperation(MlirContext context, MlirValue x, MlirLocation location) {
    return ::mlir::go::_createUnOp<::mlir::go::NegCOp>(context, x, location);
}

MlirOperation mlirGoCreateNegFOperation(MlirContext context, MlirValue x, MlirLocation location) {
    return ::mlir::go::_createUnOp<::mlir::go::NegFOp>(context, x, location);
}

MlirOperation mlirGoCreateNegIOperation(MlirContext context, MlirValue x, MlirLocation location) {
    return ::mlir::go::_createUnOp<::mlir::go::NegIOp>(context, x, location);
}

MlirOperation mlirGoCreateNotOperation(MlirContext context, MlirValue x, MlirLocation location) {
    return ::mlir::go::_createUnOp<::mlir::go::NotOp>(context, x, location);
}

MlirOperation mlirGoCreateRecvOperation(MlirContext context, MlirValue x, bool commaOk, MlirType resultType,
                                        MlirLocation location) {
    auto _context = unwrap(context);
    auto _x = unwrap(x);
    auto _resultType = unwrap(resultType);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::RecvOp>(_location, _resultType, _x,
                                                             commaOk
                                                                 ? mlir::UnitAttr::get(_context)
                                                                 : mlir::UnitAttr());
    return wrap(op);
}

//===----------------------------------------------------------------------===//
// Map Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateMapUpdateOperation(MlirContext context, MlirValue map, MlirValue key, MlirValue value,
                                             MlirLocation location) {
    auto _context = unwrap(context);
    auto _map = unwrap(map);
    auto _key = unwrap(key);
    auto _value = unwrap(value);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::MapUpdateOp>(_location, _map, _key, _value);
    return wrap(op);
}

MlirOperation mlirGoCreateMapLookupOperation(MlirContext context, MlirType resultType, MlirValue map, MlirValue key,
                                             bool hasOk,
                                             MlirLocation location) {
    auto _context = unwrap(context);
    auto _resultType = unwrap(resultType);
    auto _map = unwrap(map);
    auto _key = unwrap(key);
    auto _location = unwrap(location);

    auto boolType = mlir::IntegerType::get(_context, 1);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::MapLookupOp>(_location, _resultType,
                                                                  hasOk ? boolType : Type(),
                                                                  _map, _key);
    return wrap(op);
}

//===----------------------------------------------------------------------===//
// Memory Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateAllocaOperation(MlirContext context, MlirType resultType, MlirType elementType,
                                          MlirValue *numElements, bool isHeap, MlirLocation location) {
    auto _context = unwrap(context);
    auto _resultType = unwrap(resultType);
    auto _elementType = unwrap(elementType);
    auto _location = unwrap(location);
    mlir::Value _numElements;
    if (numElements) {
        _numElements = unwrap(*numElements);
    }
    auto _heap = isHeap ? UnitAttr::get(_context) : UnitAttr();

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::AllocaOp>(_location, _resultType, _elementType, _numElements,
                                                               _heap, StringAttr());
    return wrap(op);
}

void mlirGoAllocaOperationSetName(MlirOperation op, MlirStringRef name) {
    auto _op = mlir::cast<::mlir::go::AllocaOp>(unwrap(op));
    auto _name = unwrap(name);
    _op.setVarName(_name);
}

void mlirGoAllocaOperationSetIsHeap(MlirOperation op, bool isHeap) {
    auto _op = mlir::cast<::mlir::go::AllocaOp>(unwrap(op));
    _op.setHeap(isHeap);
}

MlirOperation mlirGoCreateLoadOperation(MlirContext context, MlirValue x, MlirType resultType, MlirLocation location) {
    auto _context = unwrap(context);
    auto _x = unwrap(x);
    auto _resultType = unwrap(resultType);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::LoadOp>(_location, _resultType, _x);
    return wrap(op);
}

MlirOperation mlirGoCreateVolatileLoadOperation(MlirContext context, MlirValue x, MlirType resultType,
                                                MlirLocation location) {
    auto _context = unwrap(context);
    auto _x = unwrap(x);
    auto _resultType = unwrap(resultType);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op =
            builder.create<::mlir::go::LoadOp>(_location, _resultType, _x, mlir::UnitAttr::get(_context),
                                               mlir::UnitAttr());
    return wrap(op);
}

MlirOperation mlirGoCreateAtomicLoadOperation(MlirContext context, MlirValue x, MlirType resultType,
                                              MlirLocation location) {
    auto _context = unwrap(context);
    auto _x = unwrap(x);
    auto _resultType = unwrap(resultType);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op =
            builder.create<::mlir::go::LoadOp>(_location, _resultType, _x, mlir::UnitAttr(),
                                               mlir::UnitAttr::get(_context));
    return wrap(op);
}

MlirOperation mlirGoCreateGepOperation(MlirContext context, MlirValue addr, MlirType baseType, intptr_t nConstIndices,
                                       int32_t *constIndices, intptr_t nDynamicIndices, MlirValue *dynamicIndices,
                                       MlirType type, MlirLocation location) {
    auto _context = unwrap(context);
    auto _addr = unwrap(addr);
    auto _baseType = unwrap(baseType);
    auto _type = unwrap(type);
    auto _location = unwrap(location);

    ::llvm::SmallVector<::mlir::Value> _dynamicIndices;
    (void) unwrapList(nDynamicIndices, dynamicIndices, _dynamicIndices);

    ::llvm::ArrayRef<int32_t> _constIndices(constIndices, nConstIndices);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op =
            builder.create<::mlir::go::GetElementPointerOp>(_location, _type, _addr, _baseType, _dynamicIndices,
                                                            _constIndices);
    return wrap(op);
}

MlirOperation mlirGoCreateGlobalOperation(MlirContext context, MlirAttribute *linkage, MlirStringRef symbol,
                                          MlirType type, MlirLocation location) {
    auto _context = unwrap(context);
    auto _type = unwrap(type);
    auto _location = unwrap(location);
    auto _symbol = unwrap(symbol);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op =
            builder.create<::mlir::go::GlobalOp>(_location, _type, mlir::StringAttr::get(_context, _symbol));

    if (linkage) {
        const auto _linkage = unwrap(*linkage);
        op->setAttr("llvm.linkage", _linkage);
    }

    return wrap(op);
}

MlirOperation mlirGoCreateYieldOperation(MlirContext context, MlirValue value, MlirLocation location) {
    auto _context = unwrap(context);
    auto _value = unwrap(value);
    auto _location = unwrap(location);
    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::YieldOp>(_location, _value);
    return wrap(op);
}

MlirOperation mlirGoCreateStoreOperation(MlirContext context, MlirValue value, MlirValue address,
                                         MlirLocation location) {
    auto _context = unwrap(context);
    auto _value = unwrap(value);
    auto _address = unwrap(address);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op =
            builder.create<::mlir::go::StoreOp>(_location, _value, _address, mlir::UnitAttr(), mlir::UnitAttr());
    return wrap(op);
}

MlirOperation mlirGoCreateVolatileStoreOperation(MlirContext context, MlirValue value, MlirValue address,
                                                 MlirLocation location) {
    auto _context = unwrap(context);
    auto _value = unwrap(value);
    auto _address = unwrap(address);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op =
            builder.create<::mlir::go::StoreOp>(_location, _value, _address, mlir::UnitAttr::get(_context),
                                                mlir::UnitAttr());
    return wrap(op);
}

MlirOperation mlirGoCreateAtomicStoreOperation(MlirContext context, MlirValue value, MlirValue address,
                                               MlirLocation location) {
    auto _context = unwrap(context);
    auto _value = unwrap(value);
    auto _address = unwrap(address);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op =
            builder.create<::mlir::go::StoreOp>(_location, _value, _address, mlir::UnitAttr(),
                                                mlir::UnitAttr::get(_context));
    return wrap(op);
}

MlirOperation mlirGoCreateSliceOperation(MlirContext context, MlirValue input, MlirValue *low, MlirValue *high,
                                         MlirValue *max, MlirType resultType, MlirLocation location) {
    auto _context = unwrap(context);
    auto _input = unwrap(input);
    auto _low = low ? unwrap(*low) : mlir::Value();
    auto _high = high ? unwrap(*high) : mlir::Value();
    auto _max = max ? unwrap(*max) : mlir::Value();
    auto _resultType = unwrap(resultType);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::SliceOp>(_location, _resultType, _input, _low, _high, _max);

    auto operandSegmentSizesAttr = DenseI32ArrayAttr::get(_context, {_low ? 1 : 0, _high ? 1 : 0, _max ? 1 : 0});
    op->setAttr("operandSegmentSizes", operandSegmentSizesAttr);

    return wrap(op);
}

MlirOperation mlirGoCreateAddressOfOperation(MlirContext context, MlirStringRef symbol, MlirType resultType,
                                             MlirLocation location) {
    auto _context = unwrap(context);
    auto _resultType = unwrap(resultType);
    auto _location = unwrap(location);
    auto _symbol = unwrap(symbol);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op =
            builder.create<::mlir::go::AddressOfOp>(_location, _resultType,
                                                    mlir::FlatSymbolRefAttr::get(_context, _symbol));
    return wrap(op);
}

//===----------------------------------------------------------------------===//
// Slice Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateSliceAddrOperation(MlirContext context, MlirType resultType, MlirValue slice, MlirValue index,
                                             MlirLocation location) {
    auto _context = unwrap(context);
    auto _resultType = unwrap(resultType);
    auto _slice = unwrap(slice);
    auto _index = unwrap(index);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::SliceAddrOp>(_location, _resultType, _slice, _index);
    return wrap(op);
}

//===----------------------------------------------------------------------===//
// Aggregate Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateExtractOperation(MlirContext context, uint64_t index, MlirType fieldType,
                                           MlirValue structValue, MlirLocation location) {
    auto _context = unwrap(context);
    auto _fieldType = unwrap(fieldType);
    auto _structValue = unwrap(structValue);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::ExtractOp>(_location, _fieldType, index, _structValue);
    return wrap(op);
}

MlirOperation mlirGoCreateInsertOperation(MlirContext context, uint64_t index, MlirValue value, MlirValue structValue,
                                          MlirType structType, MlirLocation location) {
    auto _context = unwrap(context);
    auto _value = unwrap(value);
    auto _structType = unwrap(structType);
    auto _structValue = unwrap(structValue);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::InsertOp>(_location, _structType, _value, index, _structValue);
    return wrap(op);
}

//===----------------------------------------------------------------------===//
// Constant Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateConstantOperation(MlirContext context, MlirAttribute value, MlirType type,
                                            MlirLocation location) {
    auto _context = unwrap(context);
    auto _value = unwrap(value);
    auto _type = unwrap(type);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::ConstantOp>(_location, _type, _value);
    return wrap(op);
}

//===----------------------------------------------------------------------===//
// Casting Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateBitcastOperation(MlirContext context, MlirValue value, MlirType type, MlirLocation location) {
    auto _context = unwrap(context);
    auto _value = unwrap(value);
    auto _type = unwrap(type);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::BitcastOp>(_location, _type, _value);
    return wrap(op);
}

MlirOperation mlirGoCreateComplexExtendOperation(MlirContext context, MlirValue value, MlirType type,
                                                 MlirLocation location) {
    auto _context = unwrap(context);
    auto _value = unwrap(value);
    auto _type = unwrap(type);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::ComplexExtendOp>(_location, _type, _value);
    return wrap(op);
}

MlirOperation mlirGoCreateComplexTruncateOperation(MlirContext context, MlirValue value, MlirType type,
                                                   MlirLocation location) {
    auto _context = unwrap(context);
    auto _value = unwrap(value);
    auto _type = unwrap(type);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::ComplexTruncateOp>(_location, _type, _value);
    return wrap(op);
}

MlirOperation mlirGoCreateIntToPtrOperation(MlirContext context, MlirValue value, MlirType type,
                                            MlirLocation location) {
    auto _context = unwrap(context);
    auto _value = unwrap(value);
    auto _type = unwrap(type);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::IntToPtrOp>(_location, _type, _value);
    return wrap(op);
}

MlirOperation mlirGoCreatePtrToIntOperation(MlirContext context, MlirValue value, MlirType type,
                                            MlirLocation location) {
    auto _context = unwrap(context);
    auto _value = unwrap(value);
    auto _type = unwrap(type);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::PtrToIntOp>(_location, _type, _value);
    return wrap(op);
}

MlirOperation mlirGoCreateFloatTruncateOperation(MlirContext context, MlirValue value, MlirType type,
                                                 MlirLocation location) {
    auto _context = unwrap(context);
    auto _value = unwrap(value);
    auto _type = unwrap(type);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::FloatTruncateOp>(_location, _type, _value);
    return wrap(op);
}

MlirOperation mlirGoCreateIntTruncateOperation(MlirContext context, MlirValue value, MlirType type,
                                               MlirLocation location) {
    auto _context = unwrap(context);
    auto _value = unwrap(value);
    auto _type = unwrap(type);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::IntTruncateOp>(_location, _type, _value);
    return wrap(op);
}

MlirOperation mlirGoCreateFloatExtendOperation(MlirContext context, MlirValue value, MlirType type,
                                               MlirLocation location) {
    auto _context = unwrap(context);
    auto _value = unwrap(value);
    auto _type = unwrap(type);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::FloatExtendOp>(_location, _type, _value);
    return wrap(op);
}

MlirOperation mlirGoCreateSignedExtendOperation(MlirContext context, MlirValue value, MlirType type,
                                                MlirLocation location) {
    auto _context = unwrap(context);
    auto _value = unwrap(value);
    auto _type = unwrap(type);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::SignedExtendOp>(_location, _type, _value);
    return wrap(op);
}

MlirOperation mlirGoCreateZeroExtendOperation(MlirContext context, MlirValue value, MlirType type,
                                              MlirLocation location) {
    auto _context = unwrap(context);
    auto _value = unwrap(value);
    auto _type = unwrap(type);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::ZeroExtendOp>(_location, _type, _value);
    return wrap(op);
}

MlirOperation mlirGoCreateFloatToUnsignedIntOperation(MlirContext context, MlirValue value, MlirType type,
                                                      MlirLocation location) {
    auto _context = unwrap(context);
    auto _value = unwrap(value);
    auto _type = unwrap(type);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::FloatToUnsignedIntOp>(_location, _type, _value);
    return wrap(op);
}

MlirOperation mlirGoCreateFloatToSignedIntOperation(MlirContext context, MlirValue value, MlirType type,
                                                    MlirLocation location) {
    auto _context = unwrap(context);
    auto _value = unwrap(value);
    auto _type = unwrap(type);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::FloatToSignedIntOp>(_location, _type, _value);
    return wrap(op);
}

MlirOperation mlirGoCreateUnsignedIntToFloatOperation(MlirContext context, MlirValue value, MlirType type,
                                                      MlirLocation location) {
    auto _context = unwrap(context);
    auto _value = unwrap(value);
    auto _type = unwrap(type);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::UnsignedIntToFloatOp>(_location, _type, _value);
    return wrap(op);
}

MlirOperation mlirGoCreateSignedIntToFloatOperation(MlirContext context, MlirValue value, MlirType type,
                                                    MlirLocation location) {
    auto _context = unwrap(context);
    auto _value = unwrap(value);
    auto _type = unwrap(type);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::SignedIntToFloatOp>(_location, _type, _value);
    return wrap(op);
}

MlirOperation mlirGoCreateFunctionToPointerOperation(MlirContext context, MlirValue value, MlirType type,
                                                     MlirLocation location) {
    auto _context = unwrap(context);
    auto _value = unwrap(value);
    auto _type = unwrap(type);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::FunctionToPointerOp>(_location, _type, _value);
    return wrap(op);
}

MlirOperation mlirGoCreatePointerToFunctionOperation(MlirContext context, MlirValue value, MlirType type,
                                                     MlirLocation location) {
    auto _context = unwrap(context);
    auto _value = unwrap(value);
    auto _type = unwrap(type);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::PointerToFunctionOp>(_location, _type, _value);
    return wrap(op);
}

MlirOperation mlirGoCreateChangeInterfaceOperation(MlirContext context, MlirValue value, MlirType type,
                                                   MlirLocation location) {
    auto _context = unwrap(context);
    auto _value = unwrap(value);
    auto _type = unwrap(type);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::ChangeInterfaceOp>(_location, _type, _value);
    return wrap(op);
}

MlirOperation mlirGoCreateTypeAssertOperation(MlirContext context, MlirValue value, MlirType type,
                                              MlirLocation location) {
    auto _context = unwrap(context);
    auto _value = unwrap(value);
    auto _type = unwrap(type);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::TypeAssertOp>(_location,
                                                                   _type,
                                                                   ::mlir::IntegerType::get(_context, 1),
                                                                   _value);
    return wrap(op);
}

//===----------------------------------------------------------------------===//
// Function Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoGetFunction(MlirContext context, MlirStringRef symbol, MlirType type, MlirLocation location) {
    auto _context = unwrap(context);
    auto _symbol = unwrap(symbol);
    auto _type = unwrap(type);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op =
            builder.create<::mlir::func::ConstantOp>(_location, _type, ::mlir::SymbolRefAttr::get(_context, _symbol));
    return wrap(op);
}

//===----------------------------------------------------------------------===//
// Intrinsic Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateDeclareTypeOperation(MlirContext context, MlirType type, MlirAttribute attributes,
                                               MlirLocation location) {
    auto _context = unwrap(context);
    auto _type = unwrap(type);
    auto _attributes = mlir::cast<DictionaryAttr>(unwrap(attributes));
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::DeclareTypeOp>(_location, _type);
    op->setAttrs(_attributes);
    return wrap(op);
}

MlirOperation mlirGoCreateTypeInfoOperation(MlirContext context, MlirType resultType, MlirType type,
                                            MlirLocation location) {
    auto _context = unwrap(context);
    auto _resultType = unwrap(resultType);
    auto _type = unwrap(type);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::TypeInfoOp>(_location, _resultType, _type);
    return wrap(op);
}

//===----------------------------------------------------------------------===//
// Builtin Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreatePanicOperation(MlirContext context, MlirValue value, MlirBlock *recoverBlock,
                                         MlirLocation location) {
    auto _context = unwrap(context);
    auto _location = unwrap(location);
    auto _value = unwrap(value);

    mlir::SmallVector<mlir::Block *> successors{};
    if (recoverBlock) {
        auto _recoverBlock = unwrap(*recoverBlock);
        successors.push_back(_recoverBlock);
    }

    mlir::Operation *op;
    mlir::OpBuilder builder(_context);
    op = builder.create<::mlir::go::PanicOp>(_location, _value, successors);
    return wrap(op);
}

MlirOperation mlirGoCreateRecoverOperation(MlirContext context, MlirType type, MlirLocation location) {
    auto _context = unwrap(context);
    auto _type = unwrap(type);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::RecoverOp>(_location, _type);
    return wrap(op);
}

//===----------------------------------------------------------------------===//
// Atomic Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateAtomicAddIOperation(MlirContext context, MlirType resultType, MlirValue addr, MlirValue delta,
                                              MlirLocation location) {
    return ::mlir::go::_createBinOp<::mlir::go::AtomicAddIOp>(context, resultType, addr, delta, location);
}

MlirOperation mlirGoCreateAtomicCompareAndSwapOperation(MlirContext context, MlirType resultType, MlirValue addr,
                                                        MlirValue old, MlirValue value, MlirLocation location) {
    auto _context = unwrap(context);
    auto _resultType = unwrap(resultType);
    auto _addr = unwrap(addr);
    auto _old = unwrap(old);
    auto _value = unwrap(value);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op =
            builder.create<::mlir::go::AtomicCompareAndSwapIOp>(_location, _resultType, _addr, _old, _value);
    return wrap(op);
}

MlirOperation mlirGoCreateAtomicSwapOperation(MlirContext context, MlirType resultType, MlirValue addr, MlirValue value,
                                              MlirLocation location) {
    return ::mlir::go::_createBinOp<::mlir::go::AtomicSwapIOp>(context, resultType, addr, value, location);
}

//===----------------------------------------------------------------------===//
// Control Flow Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateBranchOperation(MlirContext context, MlirBlock dest, intptr_t nDestOperands,
                                          MlirValue *destOperands, MlirLocation location) {
    auto _context = unwrap(context);
    auto _dest = unwrap(dest);
    auto _location = unwrap(location);

    ::llvm::SmallVector<::mlir::Value> _destOperands;
    (void) unwrapList(nDestOperands, destOperands, _destOperands);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::BranchOp>(_location, _destOperands, _dest);
    return wrap(op);
}

MlirOperation mlirGoCreateCondBranchOperation(MlirContext context, MlirValue condition, MlirBlock trueDest,
                                              intptr_t nTrueDestOperands, MlirValue *trueDestOperands,
                                              MlirBlock falseDest, intptr_t nFalseDestOperands,
                                              MlirValue *falseDestOperands, MlirLocation location) {
    auto _context = unwrap(context);
    auto _condition = unwrap(condition);
    auto _trueDest = unwrap(trueDest);
    auto _falseDest = unwrap(falseDest);
    auto _location = unwrap(location);

    ::llvm::SmallVector<::mlir::Value> _trueDestOperands;
    (void) unwrapList(nTrueDestOperands, trueDestOperands, _trueDestOperands);

    ::llvm::SmallVector<::mlir::Value> _falseDestOperands;
    (void) unwrapList(nFalseDestOperands, falseDestOperands, _falseDestOperands);

    auto operandSegmentSizesAttr =
            DenseI32ArrayAttr::get(_context, {
                                       1, static_cast<int>(nTrueDestOperands), static_cast<int>(nFalseDestOperands)
                                   });

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::CondBranchOp>(_location, _condition, _trueDestOperands,
                                                                   _falseDestOperands, _trueDest, _falseDest);
    op->setAttr("operandSegmentSizes", operandSegmentSizesAttr);
    return wrap(op);
}

MlirOperation mlirGoCreateReturnOperation(MlirContext context, intptr_t nOperands, MlirValue *operands,
                                          MlirLocation location) {
    auto _context = unwrap(context);
    auto _location = unwrap(location);

    ::llvm::SmallVector<::mlir::Value> _operands;
    (void) unwrapList(nOperands, operands, _operands);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::func::ReturnOp>(_location, _operands);
    return wrap(op);
}

//===----------------------------------------------------------------------===//
// Call Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateCallOperation(MlirContext context, MlirStringRef callee, intptr_t nResultTypes,
                                        MlirType *resultTypes, intptr_t nOperands, MlirValue *operands,
                                        MlirLocation location) {
    auto _context = unwrap(context);
    auto _callee = unwrap(callee);
    auto _location = unwrap(location);

    ::llvm::SmallVector<::mlir::Type> _resultTypes;
    (void) unwrapList(nResultTypes, resultTypes, _resultTypes);

    ::llvm::SmallVector<::mlir::Value> _operands;
    (void) unwrapList(nOperands, operands, _operands);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::func::CallOp>(_location, _resultTypes, _callee, _operands);
    return wrap(op);
}

MlirOperation mlirGoCreateCallIndirectOperation(MlirContext context, MlirValue callee, intptr_t nResultTypes,
                                                MlirType *resultTypes, intptr_t nOperands, MlirValue *operands,
                                                MlirLocation location) {
    auto _context = unwrap(context);
    auto _callee = unwrap(callee);
    auto _location = unwrap(location);

    ::llvm::SmallVector<::mlir::Type> _resultTypes;
    (void) unwrapList(nResultTypes, resultTypes, _resultTypes);

    ::llvm::SmallVector<::mlir::Value> _operands;
    (void) unwrapList(nOperands, operands, _operands);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::CallIndirectOp>(_location, _resultTypes, _callee, _operands);
    return wrap(op);
}

MlirOperation mlirGoCreateDeferOperation(MlirContext context, MlirValue fn, MlirAttribute *method, intptr_t nArgs,
                                         MlirValue *args, MlirLocation location) {
    auto _context = unwrap(context);
    auto _fn = unwrap(fn);
    StringAttr _method;
    if (method != nullptr) {
        _method = mlir::cast<StringAttr>(unwrap(*method));
    }
    auto _location = unwrap(location);

    ::llvm::SmallVector<::mlir::Value> _args;
    (void) unwrapList(nArgs, args, _args);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::DeferOp>(_location, _fn, _method, _args);
    return wrap(op);
}

MlirOperation mlirGoCreateGoOperation(MlirContext context, MlirValue fn, MlirType signature, MlirStringRef method,
                                      intptr_t nArgs,
                                      MlirValue *args, MlirLocation location) {
    auto _context = unwrap(context);
    auto _fn = unwrap(fn);
    auto _signature = mlir::cast<FunctionType>(unwrap(signature));

    StringAttr _method;
    const auto _methodStr = unwrap(method);
    if (!_methodStr.empty()) {
        _method = StringAttr::get(_context, _methodStr);
    }

    auto _location = unwrap(location);

    ::llvm::SmallVector<::mlir::Value> _args;
    (void) unwrapList(nArgs, args, _args);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::GoOp>(_location, _fn, _args, _signature, _method);
    return wrap(op);
}

MlirOperation mlirGoCreateInterfaceCall(MlirContext context, MlirStringRef callee, MlirType signature,
                                        MlirValue ifaceValue, intptr_t nArgs, MlirValue *args, MlirLocation location) {
    auto _context = unwrap(context);
    auto _callee = unwrap(callee);
    auto _signature = cast<mlir::FunctionType>(unwrap(signature));
    auto _ifaceValue = unwrap(ifaceValue);
    auto _location = unwrap(location);

    ::llvm::SmallVector<::mlir::Value> _args;
    (void) unwrapList(nArgs, args, _args);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op =
            builder.create<mlir::go::InterfaceCallOp>(_location, _signature.getResults(), _callee, _ifaceValue, _args);
    return wrap(op);
}

MlirOperation mlirGoCreateRuntimeCallOperation(MlirContext context, MlirStringRef callee, intptr_t nResultTypes,
                                               MlirType *resultTypes, intptr_t nOperands, MlirValue *operands,
                                               MlirLocation location) {
    auto _context = unwrap(context);
    auto _callee = unwrap(callee);
    auto _location = unwrap(location);

    ::llvm::SmallVector<::mlir::Type> _resultTypes;
    (void) unwrapList(nResultTypes, resultTypes, _resultTypes);

    ::llvm::SmallVector<::mlir::Value> _operands;
    (void) unwrapList(nOperands, operands, _operands);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::RuntimeCallOp>(_location, _resultTypes, _callee, _operands);
    return wrap(op);
}

//===----------------------------------------------------------------------===//
// Value Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirGoCreateZeroOperation(MlirContext context, MlirType type, MlirLocation location) {
    auto _context = unwrap(context);
    auto _type = unwrap(type);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::ZeroOp>(_location, _type);
    return wrap(op);
}

MlirOperation mlirGoCreateComplexOperation(MlirContext context, MlirType type, MlirValue real, MlirValue imag,
                                           MlirLocation location) {
    auto _context = unwrap(context);
    auto _type = unwrap(type);
    auto _real = unwrap(real);
    auto _imag = unwrap(imag);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::ComplexOp>(_location, _type, _real, _imag);
    return wrap(op);
}

MlirOperation mlirGoCreateImagOperation(MlirContext context, MlirType type, MlirValue value, MlirLocation location) {
    auto _context = unwrap(context);
    auto _type = unwrap(type);
    auto _value = unwrap(value);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::ImagOp>(_location, _type, _value);
    return wrap(op);
}

MlirOperation mlirGoCreateRealOperation(MlirContext context, MlirType type, MlirValue value, MlirLocation location) {
    auto _context = unwrap(context);
    auto _type = unwrap(type);
    auto _value = unwrap(value);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::RealOp>(_location, _type, _value);
    return wrap(op);
}

MlirOperation mlirGoCreateMakeChanOperation(MlirContext context, MlirType resultType, MlirValue *capacity,
                                            MlirLocation location) {
    auto _context = unwrap(context);
    auto _resultType = unwrap(resultType);
    auto _location = unwrap(location);

    mlir::Value _capacity;
    if (capacity) {
        _capacity = unwrap(*capacity);
    }

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::MakeChanOp>(_location, _resultType, _capacity);
    return wrap(op);
}

MlirOperation mlirGoCreateMakeMapOperation(MlirContext context, MlirType resultType, MlirValue *capacity,
                                           MlirLocation location) {
    auto _context = unwrap(context);
    auto _resultType = unwrap(resultType);
    auto _location = unwrap(location);

    mlir::Value _capacity;
    if (capacity) {
        _capacity = unwrap(*capacity);
    }

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::MakeMapOp>(_location, _resultType, _capacity);
    return wrap(op);
}

MlirOperation mlirGoCreateMakeSliceOperation(MlirContext context, MlirType resultType, MlirValue length,
                                             MlirValue *capacity, MlirLocation location) {
    auto _context = unwrap(context);
    auto _resultType = unwrap(resultType);
    auto _length = unwrap(length);
    auto _location = unwrap(location);

    mlir::Value _capacity;
    if (capacity) {
        _capacity = unwrap(*capacity);
    }

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::MakeSliceOp>(_location, _resultType, _length, _capacity);
    return wrap(op);
}

MlirOperation mlirGoCreateMakeInterfaceOperation(MlirContext context, MlirType resultType, MlirType type,
                                                 MlirValue value,
                                                 MlirLocation location) {
    auto _context = unwrap(context);
    auto _resultType = unwrap(resultType);
    auto _type = unwrap(type);
    auto _value = unwrap(value);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<::mlir::go::MakeInterfaceOp>(_location, _resultType, _type, _value);
    return wrap(op);
}
