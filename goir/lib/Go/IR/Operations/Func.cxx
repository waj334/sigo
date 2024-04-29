#include "Go/IR/GoDialect.h"
#include "Go/IR/GoOps.h"

namespace mlir::go {
    ::mlir::ParseResult FuncOp::parse(::mlir::OpAsmParser &p, ::mlir::OperationState &result) {
        StringAttr nameAttr;
        SmallVector<OpAsmParser::UnresolvedOperand> captureOperands;
        SmallVector<OpAsmParser::Argument> captureArgs;
        SmallVector<OpAsmParser::Argument> args;
        SmallVector<Type> resultTypes;
        ParseResult hasSymbolName;
        ParseResult hasCaptureList;

        // Functions can either start with a symbol name or begin with a capture list.
        hasSymbolName = p.parseOptionalSymbolName(nameAttr);
        if (failed(hasSymbolName)) {
            // Parse capture list.
            hasCaptureList =
                    p.parseCommaSeparatedList(OpAsmParser::Delimiter::Square, [&]() -> ParseResult {
                        OpAsmParser::UnresolvedOperand captureOperand;
                        OpAsmParser::Argument captureArg;
                        if (auto parseResult = p.parseOptionalOperand(captureOperand); parseResult.has_value()) {
                            if (failed(*parseResult)) {
                                // Capture operand is malformed.
                                return p.emitError(p.getCurrentLocation()) << "failure while parsing capture operand #"
                                                                           << captureOperands.size();
                            }

                            if (p.parseArrow()) {
                                return p.emitError(p.getCurrentLocation()) << "expected `->`";
                            }

                            if (p.parseArgument(captureArg, false, false)) {
                                // Capture argument is malformed.
                                return p.emitError(p.getCurrentLocation()) << "failure while parsing capture arg #"
                                                                           << captureArgs.size();
                            }

                            if (p.parseColon()) {
                                return p.emitError(p.getCurrentLocation()) << "expected `:`";
                            }

                            Type t;
                            if (p.parseType(t)) {
                                return p.emitError(p.getCurrentLocation()) << "expected type";
                            }

                            if (p.resolveOperand(captureOperand, t, result.operands)) {
                                return failure();
                            }

                            captureArg.type = t;
                            captureOperands.push_back(captureOperand);
                            captureArgs.push_back(captureArg);
                        }
                        return success();
                    });

            // The capture list MUST be present in the absence of a symbol name.
            if (succeeded(hasCaptureList)) {
                return failure();
            }
        }

        // Parse function arguments.
        if (failed(p.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, [&]() -> ParseResult {
            OpAsmParser::Argument arg;
            auto result = p.parseOptionalArgument(arg, true, false);
            if (result.has_value()) {
                if (failed(*result)) {
                    // Argument is malformed.
                    return p.emitError(p.getCurrentLocation()) << "failure while parsing arg #"
                                                               << args.size();
                }
                args.push_back(arg);
            }
            return success();
        }))) {
            return p.emitError(p.getCurrentLocation()) << "expected argument list";
        }

        // Parse the function result(s) if `->` follows the argument list.
        if (succeeded(p.parseOptionalArrow())) {
            // Optionally, a list of results may be present if the results begin with `(` character.
            if (succeeded(p.parseOptionalLParen())) {
                if (p.parseCommaSeparatedList([&]() -> ParseResult {
                    resultTypes.emplace_back();
                    NamedAttrList attrs;
                    if (p.parseType(resultTypes.back())) {
                        return p.emitError(p.getCurrentLocation()) << "failure while parsing result #"
                                                                   << resultTypes.size() - 1;
                    }
                    return success();
                })) {
                    return p.emitError(p.getCurrentLocation()) << "failure while parsing result list";
                }
                if (p.parseRParen()) {
                    return p.emitError(p.getCurrentLocation()) << "expected result list to terminate with `)`";
                }
            } else {
                // Parse a single return type.
                Type t;
                if (p.parseType(t)) {
                    return p.emitError(p.getCurrentLocation()) << "failure while parsing result";
                }
                resultTypes.push_back(t);
            }
        }

        // Parse function attributes.
        NamedAttrList parsedAttrs;
        if (p.parseOptionalAttrDictWithKeyword(parsedAttrs)) {
            return p.emitError(p.getCurrentLocation()) << "failure while parsing attribute dictionary";
        }
        result.attributes.append(parsedAttrs);

        // Combine all args
        SmallVector<OpAsmParser::Argument> allArgs;
        llvm::append_range(allArgs, captureArgs);
        llvm::append_range(allArgs, args);

        // Parse the optional body.
        auto body = result.addRegion();
        auto loc = p.getCurrentLocation();

        if (auto parseResult = p.parseOptionalRegion(*body, allArgs, false); parseResult.has_value()) {
            if (failed(*parseResult)) {
                return p.emitError(loc) << "failed to parse region";
            }

            // Function body must NOT be empty.
            if (body->empty()) {
                return p.emitError(loc) << "expected non-empty function body";
            }
        }

        return success();
    }

    void FuncOp::print(::mlir::OpAsmPrinter &p) {

    }

    LogicalResult FuncOp::verify() {
        return success();
    }
}