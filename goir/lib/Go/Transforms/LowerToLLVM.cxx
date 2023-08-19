#include "Go/Transforms/LowerToLLVM.h"
#include "Go/IR/GoOps.h"
#include "Go/Util.h"

#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>

namespace mlir::go {
    static SmallVector<Type, 1> getCallOpResultTypes(const LLVM::LLVMFunctionType calleeType) {
        SmallVector<Type, 1> results;
        if (const Type resultType = calleeType.getReturnType(); !isa<LLVM::LLVMVoidType>(resultType))
            results.push_back(resultType);
        return results;
    }

    static LLVM::CallOp createRuntimeCall(PatternRewriter &rewriter, const Location location,
                                          const std::string &funcName,
                                          const ArrayRef<Type> &results, const ArrayRef<Value> &args) {
        // Format the fully qualified function name
        const std::string qualifiedFuncName = "runtime." + funcName;
        auto callee = FlatSymbolRefAttr::get(rewriter.getContext(), qualifiedFuncName);

        Type returnType = LLVM::LLVMVoidType::get(rewriter.getContext());
        if (results.size() == 1) {
            returnType = results[0];
        } else if (results.size() > 1) {
            returnType = LLVM::LLVMStructType::getLiteral(rewriter.getContext(), results, false);
        }

        SmallVector<Type> argTypes(args.size());
        for (size_t i = 0; i < args.size(); ++i) {
            argTypes[i] = args[i].getType();
        }

        auto fnType = LLVM::LLVMFunctionType::get(rewriter.getContext(), returnType, argTypes, false);

        // Create and return the call
        return rewriter.create<LLVM::CallOp>(location, fnType, callee, args);
    }

    static Value createParameterPack(PatternRewriter &rewriter, const Location location, const ArrayRef<Value> &params,
                                     intptr_t &size, const DataLayout &layout, const LLVMTypeConverter *converter) {
        auto wordType = IntegerType::get(rewriter.getContext(), converter->getPointerBitwidth());

        /*
        {
            intptr_t numArgs
            intptr_t SIZEOF(ARG0)
            [ARG0]
            ...
            intptr_t SIZEOF(ARGN)
            [ARGN]
        }
        */

        // Create the context struct type
        SmallVector<Type> elementTypes = {wordType};

        // Append parameter types
        for (auto param: params) {
            elementTypes.push_back(wordType);
            elementTypes.push_back(param.getType());
        }
        auto packType = LLVM::LLVMStructType::getLiteral(rewriter.getContext(), elementTypes);

        // Create an undef of the parameter pack struct type
        Value packContainerValue = rewriter.create<LLVM::UndefOp>(location, packType);

        // Set the argument count in the parameter pack
        auto constantIntOp = rewriter.create<LLVM::ConstantOp>(location, wordType, params.size());
        packContainerValue = rewriter.create<LLVM::InsertValueOp>(location, packContainerValue,
                                                                  constantIntOp.getResult(), 0);

        // Populate the arguments parameter pack struct
        int64_t index = 1;
        for (auto param: params) {
            // Insert the size value
            const auto paramSize = layout.getTypeSize(param.getType());
            constantIntOp = rewriter.create<LLVM::ConstantOp>(location, wordType, paramSize);
            packContainerValue =
                    rewriter.create<LLVM::InsertValueOp>(location, packContainerValue, constantIntOp.getResult(),
                                                         index++);

            // Insert the argument value
            packContainerValue = rewriter.create<LLVM::InsertValueOp>(location, packContainerValue, param, index++);
        }

        // Get the allocated size of the parameter pack struct
        size = (intptr_t) layout.getTypeSize(packType);

        return packContainerValue;
    }

    namespace transforms::LLVM {
        struct AddressOfOpLowering : ConvertOpToLLVMPattern<AddressOfOp> {
            using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

            LogicalResult matchAndRewrite(AddressOfOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter) const override {
                rewriter.replaceOpWithNewOp<mlir::LLVM::AddressOfOp>(op, this->getVoidPtrType(), adaptor.getSymbol());
                return success();
            }
        };

        struct AddStrOpLowering : ConvertOpToLLVMPattern<AddStrOp> {
            using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

            LogicalResult matchAndRewrite(AddStrOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter) const override {
                const auto loc = op.getLoc();
                auto type = typeConverter->convertType(op.getType());
                auto runtimeCallOp = createRuntimeCall(rewriter, loc, "stringConcat", {type},
                                                       {adaptor.getLhs(), adaptor.getRhs()});
                rewriter.replaceOp(op, runtimeCallOp->getResults());
                return success();
            }
        };

        class AllocaOpLowering : public ConvertOpToLLVMPattern<AllocaOp> {
            using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

            LogicalResult matchAndRewrite(AllocaOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter) const override {
                OpBuilder::InsertionGuard guard(rewriter);
                auto elementType = this->typeConverter->convertType(adaptor.getElement());
                auto parentFunc = op->getParentOfType<mlir::LLVM::LLVMFuncOp>();
                Block &entryBlock = *parentFunc.getBody().begin();
                Value allocValue;

                if (adaptor.getHeap().has_value() && *adaptor.getHeap()) {
                    const Location loc = op.getLoc();
                    const auto module = op->getParentOfType<ModuleOp>();
                    const DataLayout dataLayout(module);
                    auto wordType = IntegerType::get(rewriter.getContext(), getTypeConverter()->getPointerBitwidth());

                    // Get the size of the element type
                    const auto allocationSize = dataLayout.getTypeSize(elementType);
                    Value sizeValue = rewriter.create<mlir::LLVM::ConstantOp>(loc, wordType, allocationSize);

                    if (adaptor.getNumElements()) {
                        OpBuilder::InsertionGuard guard(rewriter);

                        // Insert the mul op before the alloca op.
                        rewriter.setInsertionPoint(op);

                        // Calculate the size of the allocation.
                        auto mulOp = rewriter.create<mlir::LLVM::MulOp>(loc, wordType, sizeValue,
                                                                        adaptor.getNumElements());
                        sizeValue = mulOp->getResult(0);
                    }

                    // Create the runtime call to allocate memory on the heap
                    const auto runtimeAllocOp = createRuntimeCall(rewriter, loc, "alloc", {getVoidPtrType()},
                                                                  {sizeValue});
                    // Replace the original operation.
                    rewriter.replaceOp(op, runtimeAllocOp);
                    allocValue = runtimeAllocOp->getResult(0);
                } else {
                    const auto loc = parentFunc.getLoc();
                    if (!op->getBlock()->isEntryBlock()) {
                        // Create the alloca operation for the stack allocation in the entry block of its respective function.
                        rewriter.setInsertionPointToStart(&entryBlock);
                    }

                    Value sizeValue;
                    if (op.getNumElements()) {
                        sizeValue = adaptor.getNumElements();
                    } else {
                        // Alloca only creates a single element.
                        sizeValue =
                                rewriter.create<mlir::LLVM::ConstantOp>(loc, IntegerType::get(op->getContext(), 64), 1)
                                ->getResult(0);
                    }

                    allocValue = rewriter.replaceOpWithNewOp<mlir::LLVM::AllocaOp>(
                        op, mlir::LLVM::LLVMPointerType::get(rewriter.getContext()), elementType, sizeValue);
                    allocValue.getDefiningOp()->setLoc(parentFunc.getLoc());

                    // Move the constant operation before the alloca operation.
                    sizeValue.getDefiningOp()->moveBefore(allocValue.getDefiningOp());

                    // Zero initialize the value.
                    Value zeroValue = rewriter.create<mlir::LLVM::ZeroOp>(parentFunc.getLoc(), elementType);
                    rewriter.create<mlir::LLVM::StoreOp>(parentFunc.getLoc(), zeroValue, allocValue);
                }

                // Create debug information if set on the operation.
                if (op->hasAttr("llvm.debug.declare")) {
                    auto diLocalVarAttr = cast<mlir::LLVM::DILocalVariableAttr>(op->getAttr("llvm.debug.declare"));
                    rewriter.create<mlir::LLVM::DbgDeclareOp>(allocValue.getDefiningOp()->getLoc(), allocValue,
                                                              diLocalVarAttr,
                                                              mlir::LLVM::DIExpressionAttr());
                }

                // return success.
                return success();
            }
        };

        struct AtomicAddIOpLowering : ConvertOpToLLVMPattern<AtomicAddIOp> {
            using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

            LogicalResult matchAndRewrite(AtomicAddIOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter) const override {
                rewriter.replaceOpWithNewOp<mlir::LLVM::AtomicRMWOp>(op, mlir::LLVM::AtomicBinOp::add,
                                                                     adaptor.getAddr(),
                                                                     adaptor.getRhs(),
                                                                     mlir::LLVM::AtomicOrdering::acq_rel);
                return success();
            }
        };

        struct AtomicCompareAndSwapIOpLowering : ConvertOpToLLVMPattern<AtomicCompareAndSwapIOp> {
            using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

            LogicalResult matchAndRewrite(AtomicCompareAndSwapIOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter) const override {
                auto cmpxchg = rewriter.create<mlir::LLVM::AtomicCmpXchgOp>(
                    op.getLoc(), adaptor.getAddr(), adaptor.getOld(),
                    adaptor.getValue(), mlir::LLVM::AtomicOrdering::seq_cst,
                    mlir::LLVM::AtomicOrdering::seq_cst);

                // Extract the OK value from the result pair of the cmpxchg op
                Value ok = rewriter.create<mlir::LLVM::ExtractValueOp>(op.getLoc(), cmpxchg, 1);

                rewriter.replaceOp(op, ok);
                return success();
            }
        };

        struct AtomicSwapIOpLowering : ConvertOpToLLVMPattern<AtomicSwapIOp> {
            using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

            LogicalResult matchAndRewrite(AtomicSwapIOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter) const override {
                rewriter.replaceOpWithNewOp<mlir::LLVM::AtomicRMWOp>(op, mlir::LLVM::AtomicBinOp::xchg,
                                                                     adaptor.getAddr(),
                                                                     adaptor.getRhs(),
                                                                     mlir::LLVM::AtomicOrdering::acq_rel);
                return success();
            }
        };

        struct BitcastOpLowering : ConvertOpToLLVMPattern<BitcastOp> {
            using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

            mlir::LogicalResult matchAndRewrite(BitcastOp op, OpAdaptor adaptor,
                                                ConversionPatternRewriter &rewriter) const override {
                auto resultType = typeConverter->convertType(op.getType());
                if (resultType.isa<mlir::LLVM::LLVMStructType>() || adaptor.getValue().getType() == resultType) {
                    // Primitive to runtime type conversion.
                    rewriter.replaceOp(op, adaptor.getValue());
                } else {
                    rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(op, resultType, adaptor.getValue());
                }
                return success();
            }
        };

        struct ChangeInterfaceOpLowering : ConvertOpToLLVMPattern<ChangeInterfaceOp> {
            using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

            LogicalResult matchAndRewrite(ChangeInterfaceOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter) const override {
                const auto loc = op.getLoc();
                auto resultType = this->typeConverter->convertType(op.getType());

                // Create the type information for the interface's new type
                std::string symbol = typeInfoSymbol(op.getType());

                Value infoValue = rewriter.create<mlir::LLVM::AddressOfOp>(loc, getVoidPtrType(), symbol);

                // Alloca stack for the new value
                Value sizeValue = rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI64Type(), 1);
                Value addr = rewriter.create<mlir::LLVM::AllocaOp>(loc, getVoidPtrType(), resultType, sizeValue);

                // Get the underlying pointer value from the original interface
                Value ptrValue = rewriter.create<mlir::LLVM::ExtractValueOp>(
                    loc, getVoidPtrType(), adaptor.getValue(), 1);

                // Store the pointer value in the new interface value
                Value newValue = rewriter.create<mlir::LLVM::UndefOp>(loc, resultType);
                newValue = rewriter.create<mlir::LLVM::InsertValueOp>(loc, newValue, infoValue, 0);
                newValue = rewriter.create<mlir::LLVM::InsertValueOp>(loc, newValue, ptrValue, 1);

                // Store the new type information
                rewriter.create<mlir::LLVM::StoreOp>(loc, newValue, addr);

                // Load the value
                rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, resultType, addr);
                return success();
            }
        };

        struct ConstantOpLowering : ConvertOpToLLVMPattern<ConstantOp> {
            using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

            LogicalResult matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter) const override {
                const auto loc = op.getLoc();
                auto resultType = this->typeConverter->convertType(op.getType());

                if (go::isa<StringType>(op.getType())) {
                    const auto strAttr = mlir::dyn_cast<StringAttr>(op.getValue());
                    const auto strLen = strAttr.size();
                    const auto strHash = hash_value(strAttr.strref());
                    const std::string name = "cstr_" + std::to_string(strHash);

                    auto pointerT = this->getVoidPtrType();
                    auto runeT = rewriter.getIntegerType(8);
                    auto intT = this->getIntPtrType();
                    const auto arrayT = mlir::LLVM::LLVMArrayType::get(runeT, strLen);

                    // Get the pointer to the first character in the global string.
                    Value globalPtr = rewriter.create<mlir::LLVM::AddressOfOp>(loc, pointerT, name);
                    Value addr = rewriter.create<mlir::LLVM::GEPOp>(loc, pointerT, arrayT, globalPtr,
                                                                    ArrayRef<mlir::LLVM::GEPArg>{0, 0});

                    // Create the constant integer value representing this string's length.
                    Value lenVal = rewriter.create<mlir::LLVM::ConstantOp>(loc, this->getIntPtrType(),
                                                                           rewriter.getIntegerAttr(
                                                                               intT, strAttr.strref().size()));

                    // Create the string struct
                    mlir::Value structValue = rewriter.create<mlir::LLVM::UndefOp>(loc, resultType);
                    structValue = rewriter.create<mlir::LLVM::InsertValueOp>(loc, structValue, addr, 0);
                    structValue = rewriter.create<mlir::LLVM::InsertValueOp>(loc, structValue, lenVal, 1);

                    // Replace the original operation with the string struct value.
                    rewriter.replaceOp(op, structValue);
                    return success();
                }
                return failure();
            }
        };

        struct ZeroOpLowering : ConvertOpToLLVMPattern<ZeroOp> {
            using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

            LogicalResult
            matchAndRewrite(ZeroOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
                auto resultType = this->typeConverter->convertType(op.getType());
                rewriter.replaceOpWithNewOp<mlir::LLVM::ZeroOp>(op, resultType);
                return success();
            }
        };

        struct DeferOpLowering : ConvertOpToLLVMPattern<DeferOp> {
            using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

            LogicalResult matchAndRewrite(DeferOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter) const override {
                OpBuilder::InsertionGuard guard(rewriter);
                Location loc = op.getLoc();

                const auto module = op->getParentOfType<ModuleOp>();
                auto dataLayout = mlir::DataLayout(module);
                auto wordType = IntegerType::get(rewriter.getContext(), getTypeConverter()->getPointerBitwidth());

                // Create the parameter pack holding the arguments
                intptr_t packSize;
                auto pack = createParameterPack(rewriter, loc,
                                                llvm::SmallVector<mlir::Value>(adaptor.getCalleeOperands()),
                                                packSize, dataLayout, getTypeConverter());

                // Allocate memory on the heap to store the parameter pack into
                auto constantIntOp = rewriter.create<mlir::LLVM::ConstantOp>(loc, wordType, packSize);
                auto allocOp = createRuntimeCall(rewriter, loc, "alloc",
                                                 {mlir::LLVM::LLVMPointerType::get(rewriter.getContext())},
                                                 {constantIntOp.getResult()});

                // Store the parameter pack into the allocated memory
                rewriter.create<mlir::LLVM::StoreOp>(loc, pack, allocOp->getResult(0));

                // Create the runtime call to push the defer frame to the defer stack
                createRuntimeCall(rewriter, loc, "deferPush", {}, {adaptor.getCallee(), allocOp->getResult(0)});

                rewriter.eraseOp(op);
                return success();
            }
        };

        struct GetElementPointerOpLowering : ConvertOpToLLVMPattern<GetElementPointerOp> {
            using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

            LogicalResult matchAndRewrite(GetElementPointerOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter) const override {
                const auto loc = op.getLoc();
                const auto baseType = typeConverter->convertType(adaptor.getBaseType());
                const auto resultType = typeConverter->convertType(op.getType());
                SmallVector<mlir::LLVM::GEPArg> indices;
                for (auto index: adaptor.getConstIndices()) {
                    if (index & GetElementPointerOp::kValueFlag) {
                        index = index & GetElementPointerOp::kValueIndexMask;
                        indices.push_back(adaptor.getDynamicIndices()[index]);
                    } else {
                        indices.push_back(index);
                    }
                }

                rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(op, resultType, baseType, adaptor.getValue(), indices,
                                                               false);
                return success();
            }
        };

        struct GlobalOpLowering : ConvertOpToLLVMPattern<GlobalOp> {
            using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

            LogicalResult matchAndRewrite(GlobalOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter) const override {
                auto linkage = adaptor.getAttributes().getAs<mlir::LLVM::LinkageAttr>("llvm.linkage");
                if (!linkage) {
                    linkage = mlir::LLVM::LinkageAttr::get(this->getContext(), mlir::LLVM::Linkage::External);
                }


                auto elemT = this->typeConverter->convertType(adaptor.getGlobalType());

                // TODO: Any global that is NOT assigned a value in some function can be constant.
                auto global = rewriter.create<mlir::LLVM::GlobalOp>(op.getLoc(), elemT, false, linkage.getLinkage(),
                                                                    adaptor.getSymName(), Attribute());

                if (op->hasAttr("llvm.debug.global_expr")) {
                    // Attach debug information.
                    const auto diGlobalExprAttr = mlir::cast<mlir::LLVM::DIGlobalVariableExpressionAttr>(
                        op->getAttr("llvm.debug.global_expr"));
                    global.setDbgExprAttr(diGlobalExprAttr);
                }

                // Copy the initializer regions.
                rewriter.inlineRegionBefore(op.getRegion(),
                                            global.getRegion(),
                                            global.getRegion().end());


                // Erase the old global op.
                rewriter.eraseOp(op);
                return success();
            }
        };

        struct GlobalCtorsOpLowering : ConvertOpToLLVMPattern<GlobalCtorsOp> {
            using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

            LogicalResult matchAndRewrite(GlobalCtorsOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter) const override {
                rewriter.replaceOpWithNewOp<mlir::LLVM::GlobalCtorsOp>(op, adaptor.getCtors(), adaptor.getPriorities());
                return success();
            }
        };

        struct GoOperationLowering : ConvertOpToLLVMPattern<GoOp> {
            using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

            LogicalResult
            matchAndRewrite(GoOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
                Location loc = op.getLoc();
                const auto module = op->getParentOfType<ModuleOp>();
                const auto dataLayout = mlir::DataLayout(module);
                auto wordType = IntegerType::get(rewriter.getContext(), getTypeConverter()->getPointerBitwidth());

                // Create the parameter pack holding the arguments
                intptr_t packSize;
                auto pack = createParameterPack(rewriter, loc, llvm::SmallVector<Value>(adaptor.getCalleeOperands()),
                                                packSize,
                                                dataLayout, getTypeConverter());

                // Allocate memory on the heap to store the parameter pack into
                auto constantIntOp = rewriter.create<mlir::LLVM::ConstantOp>(loc, wordType, packSize);
                auto allocOp = createRuntimeCall(rewriter, loc, "alloc", {getVoidPtrType()},
                                                 {constantIntOp.getResult()});

                // Store the parameter pack into the allocated memory
                rewriter.create<mlir::LLVM::StoreOp>(loc, pack, allocOp->getResult(0));

                // Create the runtime call to schedule this function call
                createRuntimeCall(rewriter, loc, "addTask", {}, {adaptor.getCallee(), allocOp->getResult(0)});

                rewriter.eraseOp(op);
                return success();
            }
        };

        struct IntToPtrOpLowering : ConvertOpToLLVMPattern<IntToPtrOp> {
            using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

            LogicalResult matchAndRewrite(IntToPtrOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter) const override {
                auto type = typeConverter->convertType(op.getType());
                rewriter.replaceOpWithNewOp<mlir::LLVM::IntToPtrOp>(op, type, adaptor.getValue());
                return success();
            }
        };

        struct InterfaceCallOpLowering : ConvertOpToLLVMPattern<InterfaceCallOp> {
            using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

            LogicalResult matchAndRewrite(InterfaceCallOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter) const override {
                const auto loc = op.getLoc();
                auto ptrType = this->getVoidPtrType();
                auto ifaceValue = adaptor.getIface();

                SmallVector<mlir::Type> argTypes = {ptrType};
                SmallVector<mlir::Type> resultTypes;

                // Compute method hash (method name, args types, result types)
                const auto signature = cast<InterfaceType>(op.getIface().getType()).getMethods().at(
                    adaptor.getCallee().str());
                auto methodHash = computeMethodHash(adaptor.getCallee(), signature, true);

                for (size_t i = 0; i < adaptor.getCalleeOperands().size(); ++i) {
                    auto arg = adaptor.getCalleeOperands()[i];
                    argTypes.push_back(this->typeConverter->convertType(arg.getType()));
                }

                for (auto result: op->getResults()) {
                    resultTypes.push_back(this->typeConverter->convertType(result.getType()));
                }

                // Create a constant int value for the hash
                auto constantIntOp = rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI32Type(), methodHash);
                auto constHashValue = constantIntOp.getResult();

                // Perform vtable lookup
                auto runtimeCallOp =
                        createRuntimeCall(rewriter, loc, "interfaceLookUp", {ptrType, ptrType},
                                          {ifaceValue, constHashValue});

                Value receiverValue = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, runtimeCallOp->getResult(0), 0);
                Value fnPtrValue = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, runtimeCallOp->getResult(0), 1);

                // Collect the call arguments
                SmallVector<mlir::Value> callArgs = {receiverValue};
                for (auto arg: adaptor.getCalleeOperands()) {
                    callArgs.push_back(arg);
                }

                // Create the function type
                TypeConverter::SignatureConversion result(argTypes.size());
                auto fnType = FunctionType::get(rewriter.getContext(), argTypes, resultTypes);
                auto llvmFnType = mlir::dyn_cast<mlir::LLVM::LLVMFunctionType>(
                    this->getTypeConverter()->convertFunctionSignature(
                        fnType, false, getTypeConverter()->getOptions().useBarePtrCallConv, result));

                // Perform indirect call
                SmallVector<Value> operands;
                operands.reserve(1 + callArgs.size());
                operands.push_back(fnPtrValue);
                append_range(operands, callArgs);

                rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
                    op, getCallOpResultTypes(llvmFnType), mlir::TypeAttr::get(llvmFnType), FlatSymbolRefAttr(),
                    operands,
                    /*fastmathFlags=*/nullptr, /*branch_weights=*/nullptr,
                    mlir::LLVM::CConvAttr::get(rewriter.getContext(), mlir::LLVM::CConv::C),
                    /*access_groups=*/nullptr, /*alias_scopes=*/nullptr,
                    /*noalias_scopes=*/nullptr, /*tbaa=*/nullptr);

                return success();
            }
        };

        struct LoadOpLowering : ConvertOpToLLVMPattern<LoadOp> {
            using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

            mlir::LogicalResult matchAndRewrite(LoadOp op, OpAdaptor adaptor,
                                                ConversionPatternRewriter &rewriter) const override {
                auto type = typeConverter->convertType(op.getType());
                const auto module = op->getParentOfType<ModuleOp>();
                const auto layout = llvm::DataLayout(module->getAttr("llvm.data_layout").dyn_cast<StringAttr>());
                intptr_t alignment = static_cast<intptr_t>(layout.getPointerPrefAlignment().value());

                bool isVolatile = false;
                if (adaptor.getIsVolatile()) {
                    isVolatile = *adaptor.getIsVolatile();
                }

                mlir::LLVM::AtomicOrdering ordering = mlir::LLVM::AtomicOrdering::not_atomic;
                if (adaptor.getIsAtomic() && *adaptor.getIsAtomic()) {
                    ordering = mlir::LLVM::AtomicOrdering::acquire;
                }

                auto operand = adaptor.getOperand();
                rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, type, operand, alignment, isVolatile, false, false,
                                                                ordering);
                return success();
            }
        };

        struct MakeOpLowering : ConvertOpToLLVMPattern<MakeOp> {
            using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

            LogicalResult
            matchAndRewrite(MakeOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
                const auto loc = op.getLoc();
                auto baseType = underlyingType(op.getType());
                auto type = typeConverter->convertType(op.getType());
                Operation *runtimeCall;

                if (mlir::isa<ChanType>(baseType)) {
                    SmallVector<Value> args;
                    if (op->getNumOperands() > 0) {
                        args = adaptor.getOperands();
                    } else {
                        auto constantIntOp =
                                rewriter.create<mlir::LLVM::ConstantOp>(
                                    loc, IntegerType::get(rewriter.getContext(), 32), 0);
                        args = {constantIntOp.getResult()};
                    }
                    runtimeCall = createRuntimeCall(rewriter, loc, "chanMake", type, args);
                } else if (mlir::isa<SliceType>(baseType)) {
                    SmallVector<Value> args = adaptor.getOperands();
                    runtimeCall = createRuntimeCall(rewriter, loc, "sliceMake", type, args);
                } else if (mlir::isa<MapType>(baseType)) {
                    SmallVector<Value> args;
                    if (op->getNumOperands() > 0) {
                        args = adaptor.getOperands();
                    } else {
                        auto constantIntOp =
                                rewriter.create<mlir::LLVM::ConstantOp>(
                                    loc, IntegerType::get(rewriter.getContext(), 32), 0);
                        args = {constantIntOp.getResult()};
                    }
                    runtimeCall = createRuntimeCall(rewriter, loc, "mapMake", type, args);
                } else if (mlir::isa<InterfaceType>(baseType)) {
                    SmallVector<Value> args = adaptor.getOperands();
                    runtimeCall = createRuntimeCall(rewriter, loc, "interfaceMake", type, args);
                } else {
                    return failure();
                }

                rewriter.replaceOp(op, runtimeCall->getResults());
                return success();
            }
        };

        struct PanicOpLowering : ConvertOpToLLVMPattern<PanicOp> {
            using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

            LogicalResult matchAndRewrite(PanicOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter) const override {
                const auto loc = op.getLoc();

                // Create the runtime call to schedule this function call
                createRuntimeCall(rewriter, loc, "_panic", {}, {adaptor.getValue()});

                // The panic operation may or may not branch to the parent function's recover block if it exists.
                if (op->hasSuccessors()) {
                    // Branch to the recover block
                    rewriter.replaceOpWithNewOp<mlir::LLVM::BrOp>(op, op->getSuccessor(0));
                } else {
                    SmallVector<Type> resultTypes;
                    if (failed(this->typeConverter->convertTypes(op->getParentOp()->getResultTypes(), resultTypes))) {
                        return failure();
                    }

                    // The end of the function should be unreachable
                    rewriter.replaceOpWithNewOp<mlir::LLVM::UnreachableOp>(op, resultTypes);
                }
                return success();
            }
        };

        struct PointerToFunctionOpLowering : ConvertOpToLLVMPattern<PointerToFunctionOp> {
            using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

            LogicalResult matchAndRewrite(PointerToFunctionOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter) const override {
                auto type = typeConverter->convertType(op.getType());
                rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(op, type, adaptor.getValue());
                return success();
            }
        };

        struct PtrToIntOpLowering : ConvertOpToLLVMPattern<PtrToIntOp> {
            using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

            LogicalResult matchAndRewrite(PtrToIntOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter) const override {
                auto type = typeConverter->convertType(op.getResult().getType());
                rewriter.replaceOpWithNewOp<mlir::LLVM::PtrToIntOp>(op, type, adaptor.getValue());
                return success();
            }
        };

        struct RecoverOpLowering : ConvertOpToLLVMPattern<RecoverOp> {
            using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

            LogicalResult matchAndRewrite(RecoverOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter) const override {
                const auto loc = op.getLoc();
                auto runtimeCall = createRuntimeCall(rewriter, loc, "_recover", {}, {});
                rewriter.replaceOp(op, runtimeCall->getResults());
                return success();
            }
        };

        struct RecvOpLowering : ConvertOpToLLVMPattern<RecvOp> {
            using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

            LogicalResult
            matchAndRewrite(RecvOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
                const auto loc = op.getLoc();
                auto type = typeConverter->convertType(op.getType());
                // Allocate stack to receive the value into
                auto arrSizeConstOp = rewriter.create<mlir::LLVM::ConstantOp>(
                    loc, IntegerType::get(rewriter.getContext(), 64), 1);
                auto allocaOp = rewriter.create<mlir::LLVM::AllocaOp>(loc, type, arrSizeConstOp.getResult());

                // Create the runtime call to receive a value from the channel
                auto blockConstOp = rewriter.create<mlir::LLVM::ConstantOp>(
                    loc, IntegerType::get(rewriter.getContext(), 64),
                    adaptor.getCommaOk() ? 1 : 0);
                createRuntimeCall(rewriter, loc, "_channelReceive", {IntegerType::get(rewriter.getContext(), 1)},
                                  {
                                      op.getOperand(),
                                      allocaOp.getResult(),
                                      blockConstOp.getResult(),
                                  });
                // Load the value
                auto loadOp = rewriter.create<mlir::LLVM::LoadOp>(loc, type, allocaOp.getResult());
                rewriter.replaceOp(op, loadOp->getResults());
                return success();
            }
        };

        struct RuntimeCallOpLowering : ConvertOpToLLVMPattern<RuntimeCallOp> {
            using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

            LogicalResult matchAndRewrite(RuntimeCallOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter) const override {
                Type packedResult = nullptr;
                unsigned numResults = op.getNumResults();
                auto resultTypes = llvm::to_vector<4>(op.getResultTypes());
                auto useBarePtrCallConv = getTypeConverter()->getOptions().useBarePtrCallConv;

                if (numResults != 0) {
                    if (!(packedResult = this->getTypeConverter()->
                          packFunctionResults(resultTypes, useBarePtrCallConv)))
                        return failure();
                }

                auto callOp = rewriter.create<mlir::LLVM::CallOp>(op.getLoc(),
                                                                  packedResult ? TypeRange(packedResult) : TypeRange(),
                                                                  adaptor.getCalleeOperands(), op->getAttrs());

                SmallVector<Value, 4> results;
                if (numResults < 2) {
                    // Return directly
                    results.append(callOp.result_begin(), callOp.result_end());
                } else {
                    // Unpack result struct
                    results.reserve(numResults);
                    for (unsigned i = 0; i < numResults; ++i) {
                        results.push_back(
                            rewriter.create<mlir::LLVM::ExtractValueOp>(callOp.getLoc(), callOp->getResult(0), i));
                    }
                }
                rewriter.replaceOp(op, results);
                return success();
            }
        };

        struct SliceOpLowering : ConvertOpToLLVMPattern<SliceOp> {
            using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

            LogicalResult matchAndRewrite(SliceOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter) const override {
                Operation *resultOp;
                const auto loc = op.getLoc();
                auto baseInputType = underlyingType(op.getInput().getType());
                auto resultType = this->typeConverter->convertType(op.getType());
                auto intType = IntegerType::get(rewriter.getContext(), 32);
                if (mlir::isa<SliceType>(baseInputType)) {
                    auto symbol = typeInfoSymbol(op.getInput().getType());
                    Value typeInfoValue = rewriter.create<mlir::LLVM::AddressOfOp>(loc, getVoidPtrType(), symbol);

                    SmallVector<Value> args = {
                        adaptor.getInput(), typeInfoValue,
                        adaptor.getLow()
                            ? adaptor.getLow()
                            : rewriter.create<mlir::LLVM::ConstantOp>(loc, intType, -1).getResult(),
                        adaptor.getHigh()
                            ? adaptor.getHigh()
                            : rewriter.create<mlir::LLVM::ConstantOp>(loc, intType, -1).getResult(),
                        adaptor.getMax()
                            ? adaptor.getMax()
                            : rewriter.create<mlir::LLVM::ConstantOp>(loc, intType, -1).getResult()
                    };
                    resultOp = createRuntimeCall(rewriter, loc, "sliceReslice", {resultType}, args);
                } else if (mlir::isa<StringType>(baseInputType)) {
                    SmallVector<Value> args = {
                        adaptor.getInput(),
                        adaptor.getLow()
                            ? adaptor.getLow()
                            : rewriter.create<mlir::LLVM::ConstantOp>(loc, intType, -1).getResult(),
                        adaptor.getHigh()
                            ? adaptor.getHigh()
                            : rewriter.create<mlir::LLVM::ConstantOp>(loc, intType, -1).getResult(),
                    };
                    resultOp = createRuntimeCall(rewriter, loc, "stringSlice", {resultType}, args);
                } else if (auto ptrType = mlir::dyn_cast_or_null<PointerType>(baseInputType); ptrType) {
                    auto arrayType =
                            mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(
                                this->typeConverter->convertType(*ptrType.getElementType()));
                    const auto module = op->getParentOfType<ModuleOp>();
                    auto dataLayout = mlir::DataLayout(module);
                    auto wordType = IntegerType::get(rewriter.getContext(), getTypeConverter()->getPointerBitwidth());
                    auto stride = dataLayout.getTypeSize(arrayType.getElementType());

                    SmallVector<Value> args = {
                        adaptor.getInput(),
                        adaptor.getLow()
                            ? adaptor.getLow()
                            : rewriter.create<mlir::LLVM::ConstantOp>(loc, intType, -1).getResult(),
                        adaptor.getHigh()
                            ? adaptor.getHigh()
                            : rewriter.create<mlir::LLVM::ConstantOp>(loc, intType, -1).getResult(),
                        rewriter.create<mlir::LLVM::ConstantOp>(loc, intType, arrayType.getNumElements()).getResult(),
                        rewriter.create<mlir::LLVM::ConstantOp>(loc, wordType, stride).getResult()
                    };
                    resultOp = createRuntimeCall(rewriter, loc, "sliceAddr", {resultType}, args);
                } else {
                    return failure();
                }

                rewriter.replaceOp(op, resultOp);
                return success();
            }
        };

        struct StoreOpLowering : ConvertOpToLLVMPattern<StoreOp> {
            using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

            LogicalResult matchAndRewrite(StoreOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter) const override {
                const auto loc = op.getLoc();
                auto addr = adaptor.getAddr();
                const auto module = op->getParentOfType<ModuleOp>();
                const auto layout = llvm::DataLayout(module->getAttr("llvm.data_layout").dyn_cast<mlir::StringAttr>());
                intptr_t alignment = static_cast<intptr_t>(layout.getPointerPrefAlignment().value());

                auto value = adaptor.getValue();

                bool isVolatile = false;
                if (adaptor.getIsVolatile()) {
                    isVolatile = *adaptor.getIsVolatile();
                }

                mlir::LLVM::AtomicOrdering ordering = mlir::LLVM::AtomicOrdering::not_atomic;
                if (adaptor.getIsAtomic() && *adaptor.getIsAtomic()) {
                    ordering = mlir::LLVM::AtomicOrdering::release;
                }

                rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(op, value, addr, alignment, isVolatile, false,
                                                                 ordering);
                return success();
            }
        };

        struct ExtractOpLowering : ConvertOpToLLVMPattern<ExtractOp> {
            using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

            LogicalResult matchAndRewrite(ExtractOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter) const override {
                auto resultType = typeConverter->convertType(op.getType());
                rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(op, resultType, adaptor.getAggregate(),
                                                                        adaptor.getIndex());
                return success();
            }
        };

        struct InsertOpLowering : ConvertOpToLLVMPattern<InsertOp> {
            using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

            LogicalResult matchAndRewrite(InsertOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter) const override {
                auto resultType = typeConverter->convertType(op.getType());
                rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(op, resultType, adaptor.getAggregate(),
                                                                       adaptor.getValue(),
                                                                       adaptor.getIndex());
                return success();
            }
        };

        struct TypeInfoOpLowering : ConvertOpToLLVMPattern<TypeInfoOp> {
            using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

            LogicalResult matchAndRewrite(TypeInfoOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter) const override {
                std::string symbol = typeInfoSymbol(op.getT());
                rewriter.replaceOpWithNewOp<mlir::LLVM::AddressOfOp>(op, this->getVoidPtrType(), symbol);
                return success();
            }
        };

        struct YieldOpLowering : ConvertOpToLLVMPattern<YieldOp> {
            using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

            LogicalResult matchAndRewrite(YieldOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter) const override {
                rewriter.replaceOpWithNewOp<mlir::LLVM::ReturnOp>(op, adaptor.getInitializerValue());
                return success();
            }
        };
    } // namespace transforms::LLVM

    void populateGoToLLVMConversionPatterns(LLVMTypeConverter &converter, RewritePatternSet &patterns) {
        // clang-format off
        patterns.add<
            transforms::LLVM::AddressOfOpLowering,
            transforms::LLVM::AddStrOpLowering,
            transforms::LLVM::AllocaOpLowering,
            transforms::LLVM::AtomicAddIOpLowering,
            transforms::LLVM::AtomicCompareAndSwapIOpLowering,
            transforms::LLVM::AtomicSwapIOpLowering,
            transforms::LLVM::BitcastOpLowering,
            transforms::LLVM::ChangeInterfaceOpLowering,
            transforms::LLVM::ConstantOpLowering,
            transforms::LLVM::DeferOpLowering,
            transforms::LLVM::GetElementPointerOpLowering,
            transforms::LLVM::GlobalOpLowering,
            transforms::LLVM::GlobalCtorsOpLowering,
            transforms::LLVM::GoOperationLowering,
            transforms::LLVM::InterfaceCallOpLowering,
            transforms::LLVM::IntToPtrOpLowering,
            transforms::LLVM::LoadOpLowering,
            transforms::LLVM::MakeOpLowering,
            transforms::LLVM::PanicOpLowering,
            transforms::LLVM::PointerToFunctionOpLowering,
            transforms::LLVM::PtrToIntOpLowering,
            transforms::LLVM::RecoverOpLowering,
            transforms::LLVM::RecvOpLowering,
            transforms::LLVM::RuntimeCallOpLowering,
            transforms::LLVM::SliceOpLowering,
            transforms::LLVM::StoreOpLowering,
            transforms::LLVM::ExtractOpLowering,
            transforms::LLVM::InsertOpLowering,
            transforms::LLVM::TypeInfoOpLowering,
            transforms::LLVM::YieldOpLowering,
            transforms::LLVM::ZeroOpLowering
        >(converter);
        // clang-format off
    }
} // namespace mlir::go
