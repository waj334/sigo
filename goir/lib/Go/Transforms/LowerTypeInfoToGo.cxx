#include "Go/Transforms/LowerTypeInfoToGo.h"
#include "Go/IR/GoOps.h"
#include "Go/Util.h"

#include <Go/IR/GoAttrs.h>

#include "Go/Transforms/TypeConverter.h"

#include <llvm/ADT/TypeSwitch.h>

#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Complex/IR/Complex.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Target/LLVMIR/TypeToLLVM.h>

namespace mlir::go {
struct LowerTypeInfoPass : public mlir::PassWrapper<LowerTypeInfoPass, mlir::OperationPass<mlir::ModuleOp>> {
  mlir::DenseMap<Type, GlobalOp> m_generatedTypes;
  mlir::go::LLVMTypeConverter *typeConverter;

  void runOnOperation() final {
    auto module = getOperation();

    mlir::DataLayout dataLayout(module);
    mlir::LowerToLLVMOptions options(&getContext(), dataLayout);
    if (auto dataLayoutStr = dyn_cast<StringAttr>(module->getAttr("llvm.data_layout")); dataLayoutStr) {
      llvm::DataLayout llvmDataLayout(dataLayoutStr);
      options.dataLayout = llvmDataLayout;
    }

    this->typeConverter = new mlir::go::LLVMTypeConverter(module, options);
    LLVMConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    // Create the builder
    OpBuilder builder(module.getBodyRegion());

    // Create map of type info that should be generated
    mlir::DenseMap<mlir::Type, std::tuple<mlir::Location, mlir::Type>> typesMap;
    module.walk([&](TypeInfoOp op) {
      auto T = op.getT();
      if (typesMap.find(T) == typesMap.end()) {
        typesMap.insert({T, {op.getLoc(), T}});
      }
    });

    module.walk([&](ChangeInterfaceOp op) {
      auto T = op.getType();
      if (typesMap.find(T) == typesMap.end()) {
        typesMap.insert({T, {op.getLoc(), T}});
      }
    });

    module.walk([&](InterfaceCallOp op) {
      auto T = op.getIface().getType();
      if (typesMap.find(T) == typesMap.end()) {
        typesMap.insert({T, {op.getLoc(), T}});
      }
    });

    module.walk([&](SliceOp op) {
      auto T = op.getInput().getType();
      if (typesMap.find(T) == typesMap.end()) {
        typesMap.insert({T, {op.getLoc(), T}});
      }
    });

    // Create the globals
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(module.getBody());
      for (auto [typeHash, values] : typesMap) {
        auto [location, type] = values;
        this->createTypeInfo(builder, *typeConverter, location, type);
      }
    }
  }

  StringRef getArgument() const final { return "lower-go-type-info"; }

  StringRef getDescription() const final { return "Lower Go Type Information"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<GoDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::complex::ComplexDialect>();
    registry.insert<mlir::cf::ControlFlowDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::LLVM::LLVMDialect>();
  }

private:
  GlobalOp createTypeInfo(mlir::OpBuilder &builder, const LLVMTypeConverter &converter, const mlir::Location &loc,
                          mlir::Type T) {
    MLIRContext *context = builder.getContext();

    const auto symbol = typeInfoSymbol(T);
    // Look up the global and return it if it already exists.
    if (auto op = getOperation().lookupSymbol(symbol); op) {
      if (auto globalOp = cast<GlobalOp>(op); globalOp) {
        return globalOp;
      }
    }

    // Otherwise, begin creating the global.
    const auto infoType = converter.lookupRuntimeType("type");
    auto resultOp = this->createUninitializedGlobal(builder, infoType, symbol, loc);

    // Get the current module's data layout
    DataLayout dataLayout(getOperation());

    const auto namedDataType = converter.lookupRuntimeType("namedTypeData");
    const auto funcDataType = converter.lookupRuntimeType("funcData");
    const auto interfaceMethodDataType = converter.lookupRuntimeType("interfaceMethodData");
    const auto signatureDataType = converter.lookupRuntimeType("signatureTypeData");

    const auto ptrType = PointerType::get(context, {});
    const auto funcDataPtrType = PointerType::get(context, funcDataType);
    const auto infoPtrType = PointerType::get(context, infoType);
    const auto signatureDataPtrType = PointerType::get(context, signatureDataType);
    const auto interfaceMethodPtr = PointerType::get(context, interfaceMethodDataType);

    const auto uiType = UintType::get(context);
    const auto ui8Type = IntegerType::get(context, 8, IntegerType::Unsigned);
    const auto ui16Type = IntegerType::get(context, 16, IntegerType::Unsigned);
    const auto ui32Type = IntegerType::get(context, 32, IntegerType::Unsigned);
    const auto ui64Type = IntegerType::get(context, 64, IntegerType::Unsigned);

    const auto siType = IntType::get(context);
    const auto si8Type = IntegerType::get(context, 8, IntegerType::Signed);
    const auto si16Type = IntegerType::get(context, 16, IntegerType::Signed);
    const auto si32Type = IntegerType::get(context, 32, IntegerType::Signed);
    const auto si64Type = IntegerType::get(context, 64, IntegerType::Signed);

    const GoTypeId kind = GetGoTypeId(baseType(T));
    GlobalOp globalDataOp;
    StringRef typeName;

    globalDataOp =
        llvm::TypeSwitch<Type, GlobalOp>(T)
            .Case<ArrayType>([&](ArrayType arrayType) {
              const auto dataSymbol = typeInfoSymbol(arrayType, "_array_");
              const auto arrayDataType = converter.lookupRuntimeType("arrayTypeData");

              // TODO: The metadata for this type should be nested. Get it
              createTypeInfo(builder, converter, loc, arrayType.getElementType());

              // Create the global struct
              return createGlobal(builder, arrayDataType, dataSymbol, loc, [&](OpBuilder &builder) {
                auto typeSymbol = typeInfoSymbol(arrayType.getElementType());

                mlir::Value lenVal =
                    builder.create<ConstantOp>(loc, ui16Type, builder.getI64IntegerAttr(arrayType.getLength()));
                mlir::Value typeVal = builder.create<AddressOfOp>(loc, infoPtrType, typeSymbol);

                mlir::Value dataVal = builder.create<ZeroOp>(loc, arrayDataType);
                dataVal = builder.create<InsertOp>(loc, arrayDataType, lenVal, 0, dataVal);
                dataVal = builder.create<InsertOp>(loc, arrayDataType, typeVal, 1, dataVal);
                builder.create<YieldOp>(loc, dataVal);
              });
            })
            .Case<ChanType>([&](ChanType chanType) {
              const auto dataSymbol = typeInfoSymbol(chanType, "_chan_");
              const auto chanDataType = converter.lookupRuntimeType("chan");

              // TODO: The metadata for this type should be nested. Get it
              createTypeInfo(builder, converter, loc, chanType.getElementType());

              // Create the global struct
              return createGlobal(builder, chanDataType, dataSymbol, loc, [&](OpBuilder &builder) {
                auto typeSymbol = typeInfoSymbol(chanType.getElementType());
                mlir::Value typeInfoVal = builder.create<AddressOfOp>(loc, infoPtrType, typeSymbol);
                mlir::Value dirVal = builder.create<ConstantOp>(
                    loc, ui8Type, builder.getI64IntegerAttr(static_cast<uint8_t>(chanType.getDirection())));

                mlir::Value dataVal = builder.create<ZeroOp>(loc, chanDataType);
                dataVal = builder.create<InsertOp>(loc, chanDataType, typeInfoVal, 0, dataVal);
                dataVal = builder.create<InsertOp>(loc, chanDataType, dirVal, 1, dataVal);
                builder.create<YieldOp>(loc, dataVal);
              });
            })
            .Case<FunctionType>([&](FunctionType funcType) {
              const auto dataSymbol = typeInfoSymbol(funcType, "_signature_");
              return createGlobal(builder, signatureDataType, dataSymbol, loc, [&](OpBuilder &builder) {
                // TODO: The receiver can be known contextually from the first parameter of the function signature data.
                Value receiverTypeVal = builder.create<ZeroOp>(loc, infoPtrType);

                // Create the type information for each input parameter type.
                SmallVector<Value> inputInfoValues(funcType.getNumInputs());
                for (size_t i = 0; i < funcType.getNumInputs(); ++i) {
                  auto GV = createTypeInfo(builder, converter, loc, funcType.getInput(i));
                  inputInfoValues[i] = builder.create<AddressOfOp>(loc, infoPtrType, GV.getSymName());
                }

                // Create the slice value holding the parameters.
                const auto paramsSymbol = typeInfoSymbol(funcType, "_signature_params_");
                Value paramsSliceVal = this->createSlice(builder, paramsSymbol, infoPtrType, inputInfoValues, loc);

                // Create the type information for each result parameter type.
                SmallVector<Value> resultInfoValues(funcType.getNumResults());
                for (size_t i = 0; i < funcType.getNumResults(); ++i) {
                  auto GV = createTypeInfo(builder, converter, loc, funcType.getResult(i));
                  resultInfoValues[i] = builder.create<AddressOfOp>(loc, infoPtrType, GV.getSymName());
                }

                // Create the slice value holding the parameters.
                const auto resultsSymbol = typeInfoSymbol(funcType, "_signature_results_");
                Value resultsSliceVal = this->createSlice(builder, resultsSymbol, infoPtrType, resultInfoValues, loc);

                // Create the signature data
                Value signatureDataVal = builder.create<ZeroOp>(loc, signatureDataType);
                signatureDataVal =
                    builder.create<InsertOp>(loc, signatureDataType, receiverTypeVal, 0, signatureDataVal);
                signatureDataVal =
                    builder.create<InsertOp>(loc, signatureDataType, paramsSliceVal, 1, signatureDataVal);
                signatureDataVal =
                    builder.create<InsertOp>(loc, signatureDataType, resultsSliceVal, 2, signatureDataVal);
                builder.create<YieldOp>(loc, signatureDataVal);
              });
            })
            .Case<InterfaceType>([&](InterfaceType interfaceType) {
              const auto dataSymbol = typeInfoSymbol(interfaceType, "_interface_");
              const auto methods = interfaceType.getMethods();
              GlobalOp signaturesArrGlobalOp;

              // Create the type information for the interface method signatures referenced below.
              SmallVector<Value> methodsData(methods.size());
              size_t counter = 0;
              for (auto [name, func] : methods) {
                auto signatureInfoOp = this->createTypeInfo(builder, converter, loc, func);
                const auto id = computeMethodHash(name, func, true);
                const auto interfaceMethodSymbol =
                    typeInfoSymbol(interfaceType, "_iface_method_" + name + "_" + std::to_string(id) + "_");
                auto interfaceMethodOp =
                    createGlobal(builder, interfaceMethodDataType, interfaceMethodSymbol, loc, [&](OpBuilder &builder) {
                      // Compute the id hash of this method.

                      const Value signatureInfoValue =
                          builder.create<AddressOfOp>(loc, signatureDataPtrType, signatureInfoOp.getSymName());

                      // Create constant length value.
                      Value constId = builder.create<ConstantOp>(loc, ui32Type, builder.getI64IntegerAttr(id));

                      // Create the data struct.
                      Value interfaceMethodValue = builder.create<ZeroOp>(loc, interfaceMethodDataType);
                      interfaceMethodValue =
                          builder.create<InsertOp>(loc, interfaceMethodDataType, constId, 0, interfaceMethodValue);
                      interfaceMethodValue = builder.create<InsertOp>(loc, interfaceMethodDataType, signatureInfoValue,
                                                                      1, interfaceMethodValue);
                      builder.create<YieldOp>(loc, interfaceMethodValue);
                    });

                Value methodPtrValue =
                    builder.create<AddressOfOp>(loc, interfaceMethodPtr, interfaceMethodOp.getSymName());
                methodsData[counter++] = methodPtrValue;
              }

              // Create the global struct.
              const auto sliceType = SliceType::get(context, interfaceMethodPtr);
              return createGlobal(builder, sliceType, dataSymbol, loc, [&](OpBuilder &builder) {
                // Create the global slice holding the method signatures.
                const auto methodsSymbol = typeInfoSymbol(interfaceType, "_methods_");
                auto sliceValue = this->createSlice(builder, methodsSymbol, interfaceMethodPtr, methodsData, loc);
                builder.create<YieldOp>(loc, sliceValue);
              });
            })
            .Case<NamedType>([&](NamedType namedType) {
              GlobalOp funcsDataArrGlobalOp;
              typeName = namedType.getName().getValue();
              const auto underlyingType = namedType.getUnderlying();
              const auto _namedDataType = go::cast<mlir::LLVM::LLVMStructType>(namedDataType);
              SmallVector<GlobalOp> funcDataGlobals;

              // Create the underlying type data
              auto underlyingTypeDataOp = createTypeInfo(builder, converter, loc, underlyingType);

              // Create the method data slice.
              if (namedType.getMethods().has_value()) {
                if (const auto methods = *namedType.getMethods(); !methods.empty()) {
                  // Create each function data struct
                  size_t counter = 0;
                  for (const auto &method : methods) {
                    const auto funcSymbol = cast<mlir::FlatSymbolRefAttr>(method);

                    // Look up the function in the current module.
                    auto funcOp = cast_or_null<func::FuncOp>(getOperation().lookupSymbol(funcSymbol));
                    const auto signature = funcOp.getFunctionType();

                    // Note: It is possible for some method functions to be dropped if there is no usage of them. Do not
                    // generate information for these.
                    if (!funcOp) {
                      continue;
                    }

                    const auto fnType = cast<FunctionType>(
                        funcOp->getAttrOfType<TypeAttr>(funcOp.getFunctionTypeAttrName()).getValue());

                    // Create the info for the signature
                    auto signatureInfo = this->createTypeInfo(builder, converter, loc, signature);

                    // Exclude the qualifiers from the method name.
                    auto baseMethodName = funcSymbol.getValue();
                    baseMethodName = baseMethodName.substr(baseMethodName.find_last_of(".") + 1);

                    // Compute the method hash for this struct method.
                    const auto id = computeMethodHash(baseMethodName, fnType, false);

                    // Create the global struct
                    const auto funcDataSymbol =
                        typeInfoSymbol(T, "_" + typeName.str() + "_named_method_" + baseMethodName.str() + "_" +
                                              std::to_string(id) + "_");
                    auto funcDataGlobalOp =
                        createGlobal(builder, funcDataType, funcDataSymbol, loc, [&](OpBuilder &builder) {
                          // Get the function pointer using the Go Dialect's AddressOf operation. This will be lowered
                          // during the LLVM lowering pass.
                          Value funcPtrValue = builder.create<AddressOfOp>(loc, ptrType, funcSymbol);
                          Value signatureInfoValue =
                              builder.create<AddressOfOp>(loc, infoPtrType, signatureInfo.getSymName());

                          // Create constant id value.
                          Value constId = builder.create<ConstantOp>(loc, ui32Type, builder.getI64IntegerAttr(id));

                          Value funcDataValue = builder.create<ZeroOp>(loc, funcDataType);
                          funcDataValue = builder.create<InsertOp>(loc, funcDataType, constId, 0, funcDataValue);
                          funcDataValue = builder.create<InsertOp>(loc, funcDataType, funcPtrValue, 1, funcDataValue);
                          funcDataValue =
                              builder.create<InsertOp>(loc, funcDataType, signatureInfoValue, 2, funcDataValue);
                          builder.create<YieldOp>(loc, funcDataValue);
                        });
                    funcDataGlobals.push_back(funcDataGlobalOp);
                  }
                }
              }
              // Update the actual number of methods compiled
              const size_t numMethods = funcDataGlobals.size();

              // Create the named type data
              const auto namedTypeSymbol = typeInfoSymbol(T, "_" + typeName.str() + "_");
              return createGlobal(builder, namedDataType, namedTypeSymbol, loc, [&](OpBuilder &builder) {
                const Value underlyingTypeDataValue =
                    builder.create<AddressOfOp>(loc, infoPtrType, underlyingTypeDataOp.getSymName());

                SmallVector<Value> funcDataValues(numMethods);
                for (size_t i = 0; i < numMethods; ++i) {
                  Value funcDataAddr =
                      builder.create<AddressOfOp>(loc, funcDataPtrType, funcDataGlobals[i].getSymName());
                  funcDataValues[i] = funcDataAddr;
                }

                // Create the function data slice.
                const auto funcsArraySymbol = typeInfoSymbol(T, "_" + typeName.str() + "_methods_arr_");
                Value sliceVal = createSlice(builder, funcsArraySymbol, funcDataPtrType, funcDataValues, loc);

                // Create the final named type data struct
                Value namedTypeDataValue = builder.create<ZeroOp>(loc, namedDataType);
                namedTypeDataValue =
                    builder.create<InsertOp>(loc, namedDataType, underlyingTypeDataValue, 0, namedTypeDataValue);
                namedTypeDataValue = builder.create<InsertOp>(loc, namedDataType, sliceVal, 1, namedTypeDataValue);
                builder.create<YieldOp>(loc, namedTypeDataValue);
              });
            })
            .Case<MapType>([&](MapType mapType) {
              const auto dataSymbol = typeInfoSymbol(mapType, "_map_");
              const auto mapDataType = converter.lookupRuntimeType("mapTypeData");

              // TODO: The metadata for this type should be nested. Get it
              createTypeInfo(builder, converter, loc, mapType.getKeyType());
              createTypeInfo(builder, converter, loc, mapType.getValueType());

              // Create the global struct
              return createGlobal(builder, mapDataType, dataSymbol, loc, [&](OpBuilder &builder) {
                auto keyTypeSymbol = typeInfoSymbol(mapType.getKeyType());
                auto valueTypeSymbol = typeInfoSymbol(mapType.getValueType());

                mlir::Value keyTypeInfoVal = builder.create<AddressOfOp>(loc, ptrType, keyTypeSymbol);
                mlir::Value valueTypeInfoVal = builder.create<AddressOfOp>(loc, ptrType, valueTypeSymbol);

                mlir::Value dataVal = builder.create<ZeroOp>(loc, mapDataType);
                dataVal = builder.create<InsertOp>(loc, mapDataType, keyTypeInfoVal, 0, dataVal);
                dataVal = builder.create<InsertOp>(loc, mapDataType, valueTypeInfoVal, 1, dataVal);
                builder.create<YieldOp>(loc, dataVal);
              });
            })
            .Case<SliceType>([&](SliceType sliceType) {
              return createTypeInfo(builder, converter, loc, sliceType.getElementType());
            })
            .Case<PointerType>([&](PointerType pointerType) {
              if (pointerType.getElementType()) {
                // TODO: The metadata for this type should be nested. Get it
                return createTypeInfo(builder, converter, loc, *pointerType.getElementType());
              }
              return GlobalOp();
            })
            .Default([&](auto) { return GlobalOp(); });

    // Initialize the final type info global
    assert(kind != GoTypeId::Invalid && "type id was not determined");
    initGlobal(builder, resultOp, [&]() {
      auto kindTypeStrAttr = StringAttr::get(context, "runtime.kind");
      auto runtimePkgStrAttr = StringAttr::get(context, "runtime");
      auto kindType = NamedType::get(context, ui8Type, kindTypeStrAttr, runtimePkgStrAttr, std::nullopt);
      Value kindValue = builder.create<ConstantOp>(loc, kindType, builder.getI64IntegerAttr(static_cast<uint8_t>(kind)));

      const auto dataLayoutSpec = getOperation().getDataLayoutSpec();
      uint16_t size = go::getDefaultTypeSize(T, dataLayout, dataLayoutSpec.getEntries());

      Value sizeValue = builder.create<ConstantOp>(loc, ui16Type, builder.getI64IntegerAttr(size));

      Value infoValue = builder.create<ZeroOp>(loc, infoType);
      infoValue = builder.create<InsertOp>(loc, infoType, kindValue, 0, infoValue);
      infoValue = builder.create<InsertOp>(loc, infoType, sizeValue, 1, infoValue);

      if (globalDataOp) {
        Value dataValue = builder.create<AddressOfOp>(loc, ptrType, globalDataOp.getSymName());
        infoValue = builder.create<InsertOp>(loc, infoType, dataValue, 2, infoValue);
      }

      if (!typeName.empty()) {
        Value nameStrValue = builder.create<ConstantOp>(loc, StringType::get(context), builder.getStringAttr(typeName));
        infoValue = builder.create<InsertOp>(loc, infoType, nameStrValue, 3, infoValue);
      }

      builder.create<YieldOp>(loc, infoValue);
    });

    return resultOp;
  }

  GlobalOp createGlobal(mlir::OpBuilder &builder, mlir::Type type, const std::string &symbol, const mlir::Location &loc,
                        const std::function<void(OpBuilder &)> &fn) {
    mlir::OpBuilder::InsertionGuard guard(builder);

    // Create the global struct
    builder.setInsertionPointToStart(getOperation().getBody());
    auto globalOp = builder.create<GlobalOp>(loc, type, symbol);
    globalOp->setAttr("llvm.linkage", mlir::LLVM::LinkageAttr::get(&this->getContext(), mlir::LLVM::Linkage::External));

    // Create the the initializer block
    auto initBlock = builder.createBlock(&globalOp.getInitializerRegion());
    builder.setInsertionPointToStart(initBlock);

    // Call the lambda
    fn(builder);

    return globalOp;
  }

  GlobalOp createUninitializedGlobal(mlir::OpBuilder &builder, mlir::Type type, const std::string &symbol,
                                     const mlir::Location &loc) {
    mlir::OpBuilder::InsertionGuard guard(builder);

    // Create the global struct
    builder.setInsertionPointToStart(getOperation().getBody());
    auto op = builder.create<GlobalOp>(loc, type, symbol);
    op->setAttr("llvm.linkage", mlir::LLVM::LinkageAttr::get(builder.getContext(), mlir::LLVM::Linkage::External));
    return op;
  }

  void initGlobal(mlir::OpBuilder &builder, GlobalOp globalOp, const std::function<void()> &fn) {
    mlir::OpBuilder::InsertionGuard guard(builder);

    // Create the the initializer block
    auto initBlock = builder.createBlock(&globalOp.getInitializerRegion());
    builder.setInsertionPointToStart(initBlock);

    // Call the lambda
    fn();
  }

  Value createSlice(OpBuilder &builder, const std::string &symbol, Type elementType, ValueRange values, Location loc) {
    const auto siT = IntType::get(&this->getContext());
    const auto runtimeSliceT = this->typeConverter->lookupRuntimeType("slice");
    const auto sliceT = SliceType::get(&this->getContext(), elementType);
    const auto arrT = ArrayType::get(&this->getContext(), elementType, values.size());
    const auto ptrT = PointerType::get(&this->getContext(), std::nullopt);

    if (!values.empty()) {
      // Create the global backing array for the slice.
      auto arrGV = this->createGlobal(builder, arrT, symbol, loc, [&](OpBuilder &builder) {
        Value arrV = builder.create<ZeroOp>(loc, arrT);
        uint64_t index = 0;
        for (auto value : values) {
          // Insert the value into the array.
          arrV = builder.create<InsertOp>(loc, arrT, value, index++, arrV);

          // Move the defining operation of the value into the current block before the insert operation.
          // TODO: There might be more predecessors the might need to move.
          value.getDefiningOp()->moveBefore(arrV.getDefiningOp());
        }
        builder.create<YieldOp>(loc, arrV);
      });

      // Get the address to the global backing array.
      Value arrV = builder.create<AddressOfOp>(loc, ptrT, arrGV.getSymName());

      // Create slice struct value.
      Value lenV = builder.create<ConstantOp>(loc, siT, builder.getI64IntegerAttr(values.size()));
      Value sliceV = builder.create<ZeroOp>(loc, runtimeSliceT);

      // Build the runtime slice representation value.
      sliceV = builder.create<InsertOp>(loc, runtimeSliceT, arrV, 0, sliceV);
      sliceV = builder.create<InsertOp>(loc, runtimeSliceT, lenV, 1, sliceV);
      sliceV = builder.create<InsertOp>(loc, runtimeSliceT, lenV, 2, sliceV);

      // Cast to the dialect representation of the slice value.
      sliceV = builder.create<BitcastOp>(loc, sliceT, sliceV);
      return sliceV;
    }

    // Otherwise, return the zero value of the slice.
    Value sliceV = builder.create<ZeroOp>(loc, sliceT);
    return sliceV;
  }
};

std::unique_ptr<mlir::Pass> createLowerTypeInfoPass() { return std::make_unique<LowerTypeInfoPass>(); }
} // namespace mlir::go
