#ifndef GO_ENUMS_TD
#define GO_ENUMS_TD

include "mlir/IR/EnumAttr.td"

def Go_CmpFPredicate : I64EnumAttr<
        "CmpFPredicate", "Floating-point comparison predicates",
        [
            I64EnumAttrCase<"eq", 1>,
            I64EnumAttrCase<"gt", 2>,
            I64EnumAttrCase<"ge", 3>,
            I64EnumAttrCase<"lt", 4>,
            I64EnumAttrCase<"le", 5>,
            I64EnumAttrCase<"ne", 6>
        ]> {
    let cppNamespace = "::mlir::go";
}

def Go_CmpIPredicate : I64EnumAttr<"CmpIPredicate", "",
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
        ]> {
    let cppNamespace = "::mlir::go";
}

def Go_ChanDirection : I64EnumAttr<
        "ChanDirection", "Channel direction",
        [
            I64EnumAttrCase<"SendRecv", 0>,
            I64EnumAttrCase<"SendOnly", 1>,
            I64EnumAttrCase<"RecvOnly", 2>,
        ]> {
    let cppNamespace = "::mlir::go";
}

#endif // GO_ENUMS_TD