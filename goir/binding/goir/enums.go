package goir

/*
#include <Go-c/mlir/Enums.h>
*/
import "C"
import "pkg.si-go.dev/go-mlir/mlir"

type AsmConstraintDirection C.MlirGoAsmConstraintDirection

const (
	AsmConstraintDirectionIn    AsmConstraintDirection = C.MlirGoAsmConstraintDirectionIn
	AsmConstraintDirectionOut   AsmConstraintDirection = C.MlirGoAsmConstraintDirectionOut
	AsmConstraintDirectionInOut AsmConstraintDirection = C.MlirGoAsmConstraintDirectionInOut
)

type BasicType C.mlirGoBasicType

const (
	BasicTypeComplex BasicType = C.mlirGoBasicTypeComplex
	BasicTypeBoolean BasicType = C.mlirGoBasicTypeBoolean
	BasicTypeFloat   BasicType = C.mlirGoBasicTypeFloat
	BasicTypeInteger BasicType = C.mlirGoBasicTypeInteger
	BasicTypeNil     BasicType = C.mlirGoBasicTypeNil
	BasicTypeRune    BasicType = C.mlirGoBasicTypeRune
	BasicTypeString  BasicType = C.mlirGoBasicTypeString
)

type CmpFPredicate C.mlirGoCmpFPredicate

const (
	CmpFPredicateEq CmpFPredicate = C.mlirGoCmpFPredicate_eq
	CmpFPredicateGt CmpFPredicate = C.mlirGoCmpFPredicate_gt
	CmpFPredicateGe CmpFPredicate = C.mlirGoCmpFPredicate_ge
	CmpFPredicateLt CmpFPredicate = C.mlirGoCmpFPredicate_lt
	CmpFPredicateLe CmpFPredicate = C.mlirGoCmpFPredicate_le
	CmpFPredicateNe CmpFPredicate = C.mlirGoCmpFPredicate_ne
)

func (p CmpFPredicate) Attribute(ctx mlir.Context) mlir.Attribute {
	return wrapAttribute(C.mlirGoCreateCmpFPredicate(
		unwrapContext(ctx),
		C.mlirGoCmpFPredicate(p),
	))
}

type CmpIPredicate C.mlirGoCmpIPredicate

const (
	CmpIPredicateEq  CmpIPredicate = C.mlirGoCmpIPredicate_eq
	CmpIPredicateNe  CmpIPredicate = C.mlirGoCmpIPredicate_ne
	CmpIPredicateSlt CmpIPredicate = C.mlirGoCmpIPredicate_slt
	CmpIPredicateSle CmpIPredicate = C.mlirGoCmpIPredicate_sle
	CmpIPredicateSgt CmpIPredicate = C.mlirGoCmpIPredicate_sgt
	CmpIPredicateSge CmpIPredicate = C.mlirGoCmpIPredicate_sge
	CmpIPredicateUlt CmpIPredicate = C.mlirGoCmpIPredicate_ult
	CmpIPredicateUle CmpIPredicate = C.mlirGoCmpIPredicate_ule
	CmpIPredicateUgt CmpIPredicate = C.mlirGoCmpIPredicate_ugt
	CmpIPredicateUge CmpIPredicate = C.mlirGoCmpIPredicate_uge
)

func (p CmpIPredicate) Attribute(ctx mlir.Context) mlir.Attribute {
	return wrapAttribute(C.mlirGoCreateCmpIPredicate(
		unwrapContext(ctx),
		C.mlirGoCmpIPredicate(p),
	))
}

type CmpPredicate C.mlirGoCmpPredicate

const (
	CmpPredicateEq CmpPredicate = C.mlirGoCmpPredicate_eq
	CmpPredicateNe CmpPredicate = C.mlirGoCmpPredicate_ne
)

func (p CmpPredicate) Attribute(ctx mlir.Context) mlir.Attribute {
	return wrapAttribute(C.mlirGoCreateCmpPredicate(
		unwrapContext(ctx),
		C.mlirGoCmpPredicate(p),
	))
}

type ChanDirection C.mlirGoChanDirection

const (
	ChanDirectionSendRecv ChanDirection = C.mlirGoChanDirection_SendRecv
	ChanDirectionSendOnly ChanDirection = C.mlirGoChanDirection_SendOnly
	ChanDirectionRecvOnly ChanDirection = C.mlirGoChanDirection_RecvOnly
)

func (d ChanDirection) Attribute(ctx mlir.Context) mlir.Attribute {
	return wrapAttribute(C.mlirGoCreateChanDirection(
		unwrapContext(ctx),
		C.mlirGoChanDirection(d),
	))
}

type DISubprogramFlags C.mlirDISubprogramFlags

const (
	DISubprogramFlagsVirtual        DISubprogramFlags = C.mlirDISubprogramFlags_Virtual
	DISubprogramFlagsPureVirtual    DISubprogramFlags = C.mlirDISubprogramFlags_PureVirtual
	DISubprogramFlagsLocalToUnit    DISubprogramFlags = C.mlirDISubprogramFlags_LocalToUnit
	DISubprogramFlagsDefinition     DISubprogramFlags = C.mlirDISubprogramFlags_Definition
	DISubprogramFlagsOptimized      DISubprogramFlags = C.mlirDISubprogramFlags_Optimized
	DISubprogramFlagsPure           DISubprogramFlags = C.mlirDISubprogramFlags_Pure
	DISubprogramFlagsElemental      DISubprogramFlags = C.mlirDISubprogramFlags_Elemental
	DISubprogramFlagsRecursive      DISubprogramFlags = C.mlirDISubprogramFlags_Recursive
	DISubprogramFlagsMainSubprogram DISubprogramFlags = C.mlirDISubprogramFlags_MainSubprogram
	DISubprogramFlagsDeleted        DISubprogramFlags = C.mlirDISubprogramFlags_Deleted
	DISubprogramFlagsObjCDirect     DISubprogramFlags = C.mlirDISubprogramFlags_ObjCDirect
)

type DIFlags C.mlirDIFlags

const (
	DIFlagsZero                DIFlags = C.DIFlags_Zero
	DIFlagsBit0                DIFlags = C.DIFlags_Bit0
	DIFlagsBit1                DIFlags = C.DIFlags_Bit1
	DIFlagsPrivate             DIFlags = C.DIFlags_Private
	DIFlagsProtected           DIFlags = C.DIFlags_Protected
	DIFlagsPublic              DIFlags = C.DIFlags_Public
	DIFlagsFwdDecl             DIFlags = C.DIFlags_FwdDecl
	DIFlagsAppleBlock          DIFlags = C.DIFlags_AppleBlock
	DIFlagsReservedBit4        DIFlags = C.DIFlags_ReservedBit4
	DIFlagsVirtual             DIFlags = C.DIFlags_Virtual
	DIFlagsArtificial          DIFlags = C.DIFlags_Artificial
	DIFlagsExplicit            DIFlags = C.DIFlags_Explicit
	DIFlagsPrototyped          DIFlags = C.DIFlags_Prototyped
	DIFlagsObjcClassComplete   DIFlags = C.DIFlags_ObjcClassComplete
	DIFlagsObjectPointer       DIFlags = C.DIFlags_ObjectPointer
	DIFlagsVector              DIFlags = C.DIFlags_Vector
	DIFlagsStaticMember        DIFlags = C.DIFlags_StaticMember
	DIFlagsLValueReference     DIFlags = C.DIFlags_LValueReference
	DIFlagsRValueReference     DIFlags = C.DIFlags_RValueReference
	DIFlagsExportSymbols       DIFlags = C.DIFlags_ExportSymbols
	DIFlagsSingleInheritance   DIFlags = C.DIFlags_SingleInheritance
	DIFlagsMultipleInheritance DIFlags = C.DIFlags_MultipleInheritance
	DIFlagsVirtualInheritance  DIFlags = C.DIFlags_VirtualInheritance
	DIFlagsIntroducedVirtual   DIFlags = C.DIFlags_IntroducedVirtual
	DIFlagsBitField            DIFlags = C.DIFlags_BitField
	DIFlagsNoReturn            DIFlags = C.DIFlags_NoReturn
	DIFlagsTypePassByValue     DIFlags = C.DIFlags_TypePassByValue
	DIFlagsTypePassByReference DIFlags = C.DIFlags_TypePassByReference
	DIFlagsEnumClass           DIFlags = C.DIFlags_EnumClass
	DIFlagsThunk               DIFlags = C.DIFlags_Thunk
	DIFlagsNonTrivial          DIFlags = C.DIFlags_NonTrivial
	DIFlagsBigEndian           DIFlags = C.DIFlags_BigEndian
	DIFlagsLittleEndian        DIFlags = C.DIFlags_LittleEndian
	DIFlagsAllCallsDescribed   DIFlags = C.DIFlags_AllCallsDescribed
)
