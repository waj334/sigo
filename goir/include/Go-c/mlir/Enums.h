#ifndef GO_DIALECT_ENUMS_H
#define GO_DIALECT_ENUMS_H

#include <mlir-c/IR.h>
#include <mlir-c/Support.h>
#include <mlir-c/BuiltinAttributes.h>

#ifdef __cplusplus
extern "C" {
#endif

enum mlirGoCmpFPredicate {
    mlirGoCmpFPredicate_eq = 1,
    mlirGoCmpFPredicate_gt,
    mlirGoCmpFPredicate_ge,
    mlirGoCmpFPredicate_lt,
    mlirGoCmpFPredicate_le,
    mlirGoCmpFPredicate_ne
};

enum mlirGoCmpIPredicate {
    mlirGoCmpIPredicate_eq,
    mlirGoCmpIPredicate_ne,
    mlirGoCmpIPredicate_slt,
    mlirGoCmpIPredicate_sle,
    mlirGoCmpIPredicate_sgt,
    mlirGoCmpIPredicate_sge,
    mlirGoCmpIPredicate_ult,
    mlirGoCmpIPredicate_ule,
    mlirGoCmpIPredicate_ugt,
    mlirGoCmpIPredicate_uge
};

enum mlirGoChanDirection {
    mlirGoChanDirection_SendRecv,
    mlirGoChanDirection_SendOnly,
    mlirGoChanDirection_RecvOnly
};

enum mlirDISubprogramFlags {
    mlirDISubprogramFlags_Virtual = 1,
    mlirDISubprogramFlags_PureVirtual = 2,
    mlirDISubprogramFlags_LocalToUnit = 4,
    mlirDISubprogramFlags_Definition = 8,
    mlirDISubprogramFlags_Optimized = 16,
    mlirDISubprogramFlags_Pure = 32,
    mlirDISubprogramFlags_Elemental = 64,
    mlirDISubprogramFlags_Recursive = 128,
    mlirDISubprogramFlags_MainSubprogram = 256,
    mlirDISubprogramFlags_Deleted = 512,
    mlirDISubprogramFlags_ObjCDirect = 2048,
};

enum mlirDIFlags {
    DIFlags_Zero = 0,
    DIFlags_Bit0 = 1,
    DIFlags_Bit1 = 2,
    DIFlags_Private = 1,
    DIFlags_Protected = 2,
    DIFlags_Public = 3,
    DIFlags_FwdDecl = 4,
    DIFlags_AppleBlock = 8,
    DIFlags_ReservedBit4 = 16,
    DIFlags_Virtual = 32,
    DIFlags_Artificial = 64,
    DIFlags_Explicit = 128,
    DIFlags_Prototyped = 256,
    DIFlags_ObjcClassComplete = 512,
    DIFlags_ObjectPointer = 1024,
    DIFlags_Vector = 2048,
    DIFlags_StaticMember = 4096,
    DIFlags_LValueReference = 8192,
    DIFlags_RValueReference = 16384,
    DIFlags_ExportSymbols = 32768,
    DIFlags_SingleInheritance = 65536,
    DIFlags_MultipleInheritance = 65536,
    DIFlags_VirtualInheritance = 65536,
    DIFlags_IntroducedVirtual = 262144,
    DIFlags_BitField = 524288,
    DIFlags_NoReturn = 1048576,
    DIFlags_TypePassByValue = 4194304,
    DIFlags_TypePassByReference = 8388608,
    DIFlags_EnumClass = 16777216,
    DIFlags_Thunk = 33554432,
    DIFlags_NonTrivial = 67108864,
    DIFlags_BigEndian = 134217728,
    DIFlags_LittleEndian = 268435456,
    DIFlags_AllCallsDescribed = 536870912,
};

MlirAttribute mlirGoCreateCmpFPredicate(MlirContext context, enum mlirGoCmpFPredicate predicate);
MlirAttribute mlirGoCreateCmpIPredicate(MlirContext context, enum mlirGoCmpIPredicate predicate);
MlirAttribute mlirGoCreateChanDirection(MlirContext context, enum mlirGoChanDirection direction);

#ifdef __cplusplus
}
#endif

#endif //GO_DIALECT_ENUMS_H
