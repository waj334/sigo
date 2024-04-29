
#include "Go-c/mlir/Enums.h"
#include "Go/IR/GoDialect.h"

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Support.h>

MlirAttribute mlirGoCreateCmpFPredicate(MlirContext context, mlirGoCmpFPredicate predicate) {
    ::mlir::go::CmpFPredicate _predicate;
    switch (predicate) {
        case mlirGoCmpFPredicate_eq:
            _predicate = ::mlir::go::CmpFPredicate::eq;
            break;
        case mlirGoCmpFPredicate_gt:
            _predicate = ::mlir::go::CmpFPredicate::gt;
            break;
        case mlirGoCmpFPredicate_ge:
            _predicate = ::mlir::go::CmpFPredicate::ge;
            break;
        case mlirGoCmpFPredicate_lt:
            _predicate = ::mlir::go::CmpFPredicate::lt;
            break;
        case mlirGoCmpFPredicate_le:
            _predicate = ::mlir::go::CmpFPredicate::le;
            break;
        case mlirGoCmpFPredicate_ne:
            _predicate = ::mlir::go::CmpFPredicate::ne;
            break;
        default:
            assert(false&&"unreachable");
    }
    return wrap(::mlir::go::CmpFPredicateAttr::get(unwrap(context), _predicate));
}

MlirAttribute mlirGoCreateCmpIPredicate(MlirContext context, mlirGoCmpIPredicate predicate) {
    ::mlir::go::CmpIPredicate _predicate;
    switch (predicate) {
        case mlirGoCmpIPredicate_eq:
            _predicate = ::mlir::go::CmpIPredicate::eq;
            break;
        case mlirGoCmpIPredicate_ne:
            _predicate = ::mlir::go::CmpIPredicate::ne;
            break;
        case mlirGoCmpIPredicate_slt:
            _predicate = ::mlir::go::CmpIPredicate::slt;
            break;
        case mlirGoCmpIPredicate_sle:
            _predicate = ::mlir::go::CmpIPredicate::sle;
            break;
        case mlirGoCmpIPredicate_sgt:
            _predicate = ::mlir::go::CmpIPredicate::sgt;
            break;
        case mlirGoCmpIPredicate_sge:
            _predicate = ::mlir::go::CmpIPredicate::sge;
            break;
        case mlirGoCmpIPredicate_ult:
            _predicate = ::mlir::go::CmpIPredicate::ult;
            break;
        case mlirGoCmpIPredicate_ule:
            _predicate = ::mlir::go::CmpIPredicate::ule;
            break;
        case mlirGoCmpIPredicate_ugt:
            _predicate = ::mlir::go::CmpIPredicate::ugt;
            break;
        case mlirGoCmpIPredicate_uge:
            _predicate = ::mlir::go::CmpIPredicate::uge;
            break;
        default:
            assert(false&&"unreachable");
    }
    return wrap(::mlir::go::CmpIPredicateAttr::get(unwrap(context), _predicate));
}

MlirAttribute mlirGoCreateChanDirection(MlirContext context, mlirGoChanDirection direction) {
    ::mlir::go::ChanDirection _direction;
    switch (direction) {
        case mlirGoChanDirection_SendRecv:
            _direction = ::mlir::go::ChanDirection::SendRecv;
            break;
        case mlirGoChanDirection_SendOnly:
            _direction = ::mlir::go::ChanDirection::SendOnly;
            break;
        case mlirGoChanDirection_RecvOnly:
            _direction = ::mlir::go::ChanDirection::RecvOnly;
            break;
        default:
            assert(false&&"unreachable");
    }
    return wrap(::mlir::go::ChanDirectionAttr::get(unwrap(context), _direction));
}