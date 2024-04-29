#ifndef GO_DIALECT_TD
#define GO_DIALECT_TD

include "mlir/IR/DialectBase.td"

def Go_Dialect : Dialect {
    let name = "go";
    let summary = "A dialect implementing the Go language specification.";
    let description = [{
        This dialect implements the Go 1.19 language specification.
    }];
    let cppNamespace = "::mlir::go";
    let useDefaultTypePrinterParser = 0;
    let useDefaultAttributePrinterParser = 1;
    let extraClassDeclaration = [{
        ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;
        void printType(::mlir::Type type, ::mlir::DialectAsmPrinter &printer) const override;
    }];
}

#endif // GO_DIALECT_TD