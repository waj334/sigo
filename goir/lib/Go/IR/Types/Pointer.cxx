#include <Go/IR/GoTypes.h>
#include <mlir/IR/OpImplementation.h>

namespace mlir::go {
    ::mlir::Type PointerType::parse(::mlir::AsmParser &p) {
        std::optional<::mlir::Type> _elementType;
        if (succeeded(p.parseOptionalLess())) {
            ::mlir::Type T;
            if (p.parseType(T)) {
                p.emitError(p.getCurrentLocation(), "expected element type");
                return {};
            }
            _elementType = T;
            if (p.parseGreater()) {
                p.emitError(p.getCurrentLocation(), "expected `>`");
            }
        }
        return PointerType::get(p.getContext(), _elementType);
    }

    void PointerType::print(::mlir::AsmPrinter &p) const {
        const auto _elementType = this->getElementType();
        if (_elementType) {
            p << "<";
            p.printType(_elementType.value());
            p << ">";
        }
    }
}