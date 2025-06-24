#ifndef GO_ENUMS_TD
#define GO_ENUMS_TD

include "mlir/IR/EnumAttr.td"

class Go_I64EnumAttr<string name, string summary, list<I64EnumAttrCase> cases> : I64EnumAttr<name, summary, cases> {
    let cppNamespace = "::mlir::go";
}

class Go_EnumAttr<EnumAttrInfo enumInfo, string name = "", list <Trait> traits = []> :
        EnumAttr<Go_Dialect, enumInfo, name, traits> {
    let cppNamespace = "::mlir::go";
}

def Go_CmpFPredicate : Go_I64EnumAttr<
        "CmpFPredicate", "Floating-point comparison predicates",
        [
            I64EnumAttrCase<"eq", 1>,
            I64EnumAttrCase<"gt", 2>,
            I64EnumAttrCase<"ge", 3>,
            I64EnumAttrCase<"lt", 4>,
            I64EnumAttrCase<"le", 5>,
            I64EnumAttrCase<"ne", 6>
        ]>;

def Go_CmpIPredicate : Go_I64EnumAttr<"CmpIPredicate", "",
        [
           I64EnumAttrCase<"eq",  0>,
           I64EnumAttrCase<"ne",  1>,
           I64EnumAttrCase<"slt", 2>,
           I64EnumAttrCase<"sle", 3>,
           I64EnumAttrCase<"sgt", 4>,
           I64EnumAttrCase<"sge", 5>,
           I64EnumAttrCase<"ult", 6>,
           I64EnumAttrCase<"ule", 7>,
           I64EnumAttrCase<"ugt", 8>,
           I64EnumAttrCase<"uge", 9>
        ]>;

def Go_CmpPredicate : Go_I64EnumAttr<"CmpPredicate", "",
        [
            I64EnumAttrCase<"eq",  0>,
            I64EnumAttrCase<"ne",  1>,
        ]>;

def Go_ChanDirection : Go_I64EnumAttr<
        "ChanDirection", "Channel direction",
        [
            I64EnumAttrCase<"SendRecv", 0>,
            I64EnumAttrCase<"SendOnly", 1>,
            I64EnumAttrCase<"RecvOnly", 2>,
        ]>;

def Go_UntypedBasicKind : Go_I64EnumAttr<"UntypedBasicKind", "Untyped Basic Kind",
        [
            I64EnumAttrCase<"Complex", 0>,
            I64EnumAttrCase<"Boolean", 1>,
            I64EnumAttrCase<"Float", 2>,
            I64EnumAttrCase<"Integer", 3>,
            I64EnumAttrCase<"Nil", 4>,
            I64EnumAttrCase<"Rune", 5>,
            I64EnumAttrCase<"String", 6>,
        ]> {
    let genSpecializedAttr = 0;
}

def Go_UntypedBasicKindAttr : Go_EnumAttr<Go_UntypedBasicKind, "UntypedBasicKindAttr">;

#endif // GO_ENUMS_TD