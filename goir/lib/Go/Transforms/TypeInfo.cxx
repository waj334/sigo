#include "Go/Transforms/TypeInfo.h"

#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/CommandLine.h>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>

#include "Go/Util.h"

constexpr int64_t constSizeIndex = 0;
constexpr int64_t constDataIndex = 1;
constexpr int64_t constNameIndex = 2;
constexpr int64_t constKindIndex = 3;

namespace mlir::go
{
static llvm::SmallDenseMap<mlir::Type, uint64_t> s_generatedTypeInfoMap =
  llvm::SmallDenseMap<mlir::Type, uint64_t>();
static uint64_t s_typeInfoCounter = 0;

// Note: No mutex needed here. createTypeInfo and its callers (dialect
// conversion patterns) run single-threaded within the MLIR pass manager.
uint64_t getTypeId(const mlir::Type& type)
{
  const auto it = s_generatedTypeInfoMap.find(type);
  if (it == s_generatedTypeInfoMap.end())
  {
    const auto result = s_typeInfoCounter++;
    s_generatedTypeInfoMap[type] = result;
    return result;
  }
  return it->getSecond();
}

std::string typeInfoSymbol(const mlir::Type& type, const std::string& prefix)
{
  const auto id = getTypeId(type);
  std::string symbol = "type";
  if (!prefix.empty())
  {
    symbol += "_" + prefix;
  }
  symbol += "_" + std::to_string(id);
  return symbol;
}

::mlir::LLVM::GlobalOp createUninitializedGlobal(
  mlir::OpBuilder& builder,
  mlir::ModuleOp module,
  mlir::Type type,
  const std::string& symbol,
  const mlir::Location& loc)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  return ::mlir::LLVM::GlobalOp::create(builder, 
    loc, type, true, mlir::LLVM::Linkage::External, symbol, Attribute());
}

::mlir::LLVM::GlobalOp createGlobal(
  mlir::OpBuilder& builder,
  mlir::ModuleOp module,
  mlir::Type type,
  const std::string& symbol,
  const mlir::Location& loc,
  const std::function<void(OpBuilder&)>& fn)
{
  mlir::LLVM::GlobalOp globalOp =
    mlir::dyn_cast_or_null<mlir::LLVM::GlobalOp>(module.lookupSymbol(symbol));
  if (!globalOp)
  {
    // Create the global struct.
    globalOp = createUninitializedGlobal(builder, module, type, symbol, loc);
  }

  // This global must NOT already be initialized.
  assert(!globalOp.getInitializerBlock() || globalOp.getInitializerBlock()->empty());

  // Create the initializer block.
  mlir::OpBuilder::InsertionGuard guard(builder);
  auto initBlock = builder.createBlock(&globalOp.getInitializerRegion());
  builder.setInsertionPointToStart(initBlock);

  // Call the lambda.
  fn(builder);

  return globalOp;
}

Value createGoStringValue(
  mlir::OpBuilder& builder,
  ModuleOp module,
  const mlir::go::LLVMTypeConverter& converter,
  StringRef value,
  const mlir::Location& loc)
{
  const auto gostrType = converter.convertType(converter.lookupRuntimeType("string"));

  // Look up the C string value.
  Value cstrValue;
  const auto strHash = hash_value(llvm::StringRef(value));
  std::string cstrName = "cstr_" + std::to_string(strHash);
  if (
    auto cstrGlobalOp = mlir::dyn_cast_or_null<mlir::LLVM::GlobalOp>(module.lookupSymbol(cstrName)))
  {
    // Get the address of the existing C string.
    cstrValue = mlir::LLVM::AddressOfOp::create(builder, loc, cstrGlobalOp);
  }
  else
  {
    // Create a new global C char string to hold the string data.
    cstrValue =
      mlir::LLVM::createGlobalString(loc, builder, cstrName, value, mlir::LLVM::Linkage::External);
  }

  // Create the Go string value.
  Value gostrValue = mlir::LLVM::ZeroOp::create(builder, loc, gostrType);

  // Insert the C string address into the Go string struct value.
  gostrValue = mlir::LLVM::InsertValueOp::create(builder, loc, gostrValue, cstrValue, ArrayRef<int64_t>{0});

  // Insert the string length value into the Go string struct value return the resulting Go string
  // value.
  Value lengthValue =
    mlir::LLVM::ConstantOp::create(builder, loc, builder.getI32Type(), value.size());
  return mlir::LLVM::InsertValueOp::create(builder, loc, gostrValue, lengthValue, ArrayRef<int64_t>{1});
}

Value createSliceValue(
  mlir::OpBuilder& builder,
  ModuleOp module,
  const mlir::go::LLVMTypeConverter& converter,
  StringRef name,
  Type elementType,
  size_t length,
  const std::function<SmallVector<Value>(OpBuilder&)>& valueGeneratorFn,
  const mlir::Location& loc)
{
  const auto goSliceType = converter.convertType(converter.lookupRuntimeType("slice"));
  const std::string arrSymbol = "slice_arr_" + name.str() + "_" +
    std::to_string(reinterpret_cast<intptr_t>(elementType.getImpl()));

  // Create the global array value.
  const auto arrayType = mlir::LLVM::LLVMArrayType::get(elementType, length);
  const auto arrGlobalOp = createGlobal(
    builder,
    module,
    arrayType,
    arrSymbol,
    loc,
    [&](OpBuilder& builder)
    {
      const auto values = valueGeneratorFn(builder);
      Value arrValue = mlir::LLVM::ZeroOp::create(builder, loc, arrayType);
      for (size_t i = 0; i < length; i++)
      {
        arrValue = mlir::LLVM::InsertValueOp::create(builder, loc, arrValue, values[i], ArrayRef<int64_t>{static_cast<int64_t>(i)});
      }
      mlir::LLVM::ReturnOp::create(builder, loc, arrValue);
    });

  // Create the Go slice value.
  Value goSliceValue = mlir::LLVM::ZeroOp::create(builder, loc, goSliceType);

  // Insert the address of the backing array for the slice.
  Value arrValue = mlir::LLVM::AddressOfOp::create(builder, loc, arrGlobalOp);
  goSliceValue = mlir::LLVM::InsertValueOp::create(builder, loc, goSliceValue, arrValue, ArrayRef<int64_t>{0});

  // Insert the length and capacity values.
  Value lengthValue = mlir::LLVM::ConstantOp::create(builder, loc, builder.getI32Type(), length);
  goSliceValue = mlir::LLVM::InsertValueOp::create(builder, loc, goSliceValue, lengthValue, ArrayRef<int64_t>{1});
  goSliceValue = mlir::LLVM::InsertValueOp::create(builder, loc, goSliceValue, lengthValue, ArrayRef<int64_t>{2});
  return goSliceValue;
}

mlir::LLVM::GlobalOp createSignatureDataGlobal(
  mlir::OpBuilder& builder,
  mlir::ModuleOp module,
  const mlir::Location& loc,
  const mlir::go::FunctionType type,
  const mlir::go::LLVMTypeConverter& converter)
{
  const auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());

  // Create the chan type data.
  const auto symbol = typeInfoSymbol(type, "signature");
  if (auto globalOp = mlir::dyn_cast_or_null<mlir::LLVM::GlobalOp>(module.lookupSymbol(symbol));
      globalOp)
  {
    return globalOp;
  }

  const auto dataType = converter.convertType(converter.lookupRuntimeType("signatureTypeData"));
  return createGlobal(
    builder,
    module,
    dataType,
    symbol,
    loc,
    [&](OpBuilder& builder)
    {
      // Create func data value.
      Value dataValue = mlir::LLVM::ZeroOp::create(builder, loc, dataType);

      // Insert the receiver type data if present.
      if (const auto receiverType = type.getReceiver())
      {
        auto receiverTypeDataGlobalOp = createTypeInfo(builder, module, loc, receiverType, converter);
        Value receiverTypeDataValue =
          mlir::LLVM::AddressOfOp::create(builder, loc, receiverTypeDataGlobalOp);
        dataValue =
          mlir::LLVM::InsertValueOp::create(builder, loc, dataValue, receiverTypeDataValue, ArrayRef<int64_t>{0});
      }

      const uint64_t numInputs = type.getNumInputs();
      auto inputGeneratorFn = [&](OpBuilder& builder)
      {
        SmallVector<Value> inputTypeDataValues;
        inputTypeDataValues.reserve(numInputs);
        for (auto i = 0; i < type.getNumInputs(); i++)
        {
          // Create the type info for the input type.
          auto inputTypeInfoGlobalOp = createTypeInfo(builder, module, loc, type.getInput(i), converter);

          // Get the address of the input type info.
          Value inputTypeInfoValue =
            mlir::LLVM::AddressOfOp::create(builder, loc, inputTypeInfoGlobalOp);
          inputTypeDataValues.push_back(inputTypeInfoValue);
        }
        return inputTypeDataValues;
      };

      // Insert the slice value for the inputs type data.
      Value inputsSliceValue = createSliceValue(
        builder, module, converter, symbol + "_inputs", ptrType, numInputs, inputGeneratorFn, loc);
      dataValue = mlir::LLVM::InsertValueOp::create(builder, loc, dataValue, inputsSliceValue, ArrayRef<int64_t>{1});

      auto resultGeneratorFn = [&](OpBuilder& builder)
      {
        SmallVector<Value> resultTypeDataValues;
        resultTypeDataValues.reserve(type.getNumResults());
        for (size_t i = 0; i < type.getNumResults(); i++)
        {
          // Create the type info for the result type.
          auto resultTypeInfoGlobalOp = createTypeInfo(builder, module, loc, type.getResult(i), converter);

          // Get the address of the result type info.
          Value resultTypeInfoValue =
            mlir::LLVM::AddressOfOp::create(builder, loc, resultTypeInfoGlobalOp);
          resultTypeDataValues.push_back(resultTypeInfoValue);
        }
        return resultTypeDataValues;
      };

      // Insert the slice value for the results type data.
      Value resultsSliceValue = createSliceValue(
        builder,
        module,
        converter,
        symbol + "_results",
        ptrType,
        type.getNumResults(),
        resultGeneratorFn,
        loc);
      dataValue = mlir::LLVM::InsertValueOp::create(builder, loc, dataValue, resultsSliceValue, ArrayRef<int64_t>{2});

      // Yield the function type data.
      mlir::LLVM::ReturnOp::create(builder, loc, dataValue);
    });
}

mlir::LLVM::GlobalOp createTypeInfo(
  mlir::OpBuilder& builder,
  mlir::ModuleOp module,
  const mlir::Location& loc,
  const mlir::Type T,
  const mlir::go::LLVMTypeConverter& converter)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  const auto i8Type = builder.getI8Type();
  const auto i16Type = builder.getI16Type();
  const auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  const auto infoType = converter.convertType(converter.lookupRuntimeType("type"));
  const auto uintptrType = builder.getIntegerType(converter.getPointerBitwidth());

  // Look up type info in module first.
  const auto infoSymbol = typeInfoSymbol(T);
  if (auto globalOp = mlir::dyn_cast_or_null<mlir::LLVM::GlobalOp>(module.lookupSymbol(infoSymbol));
      globalOp)
  {
    return globalOp;
  }

  // Create the global for this type earlier to prevent infinite recursion when generating type
  // information for types.
  createUninitializedGlobal(builder, module, infoType, infoSymbol, loc);

  // Generate the type information.
  StringRef typeName;
  auto dataGlobalOp =
    TypeSwitch<Type, ::mlir::LLVM::GlobalOp>(T)
      .Case<ArrayType>(
        [&](ArrayType type)
        {
          // Create the element type data.
          auto elementTypeDataGlobalOp =
            createTypeInfo(builder, module, loc, type.getElementType(), converter);

          // Create the array type data.
          const auto symbol = typeInfoSymbol(type, "array");
          const auto arrayTypeDataType =
            converter.convertType(converter.lookupRuntimeType("arrayTypeData"));
          return createGlobal(
            builder,
            module,
            arrayTypeDataType,
            symbol,
            loc,
            [&](OpBuilder& builder)
            {
              // Create the array type data value.
              Value dataValue = mlir::LLVM::ZeroOp::create(builder, loc, arrayTypeDataType);

              // Insert the length value.
              Value lengthValue =
                mlir::LLVM::ConstantOp::create(builder, loc, builder.getI16Type(), type.getLength());
              dataValue = mlir::LLVM::InsertValueOp::create(builder, loc, dataValue, lengthValue, ArrayRef<int64_t>{0});

              // Insert the element type data pointer value.
              Value elementTypeDataValue =
                mlir::LLVM::AddressOfOp::create(builder, loc, elementTypeDataGlobalOp);
              dataValue =
                mlir::LLVM::InsertValueOp::create(builder, loc, dataValue, elementTypeDataValue, ArrayRef<int64_t>{1});

              // Yield the array type data value.
              mlir::LLVM::ReturnOp::create(builder, loc, dataValue);
            });
        })
      .Case<ChanType>(
        [&](ChanType type)
        {
          // Create the element type data.
          auto elementTypeDataGlobalOp =
            createTypeInfo(builder, module, loc, type.getElementType(), converter);

          // Create the chan type data.
          const auto symbol = typeInfoSymbol(type, "chan");
          const auto chanTypeDataType =
            converter.convertType(converter.lookupRuntimeType("channelTypeData"));
          return createGlobal(
            builder,
            module,
            chanTypeDataType,
            symbol,
            loc,
            [&](OpBuilder& builder)
            {
              // Create the chan type data value.
              Value dataValue = mlir::LLVM::ZeroOp::create(builder, loc, chanTypeDataType);

              // Insert the element type data value.
              Value elementTypeDataValue =
                mlir::LLVM::AddressOfOp::create(builder, loc, elementTypeDataGlobalOp);
              dataValue =
                mlir::LLVM::InsertValueOp::create(builder, loc, dataValue, elementTypeDataValue, ArrayRef<int64_t>{0});

              // Insert the direction value.
              Value directionVal = mlir::LLVM::ConstantOp::create(builder, 
                loc, builder.getI8Type(), static_cast<uint64_t>(type.getDirection()));
              dataValue =
                mlir::LLVM::InsertValueOp::create(builder, loc, dataValue, directionVal, ArrayRef<int64_t>{1});

              // Yield the chan type data value.
              mlir::LLVM::ReturnOp::create(builder, loc, dataValue);
            });
        })
      .Case([&](mlir::go::FunctionType type)
            { return createSignatureDataGlobal(builder, module, loc, type, converter); })
      .Case<InterfaceType>(
        [&](InterfaceType type)
        {
          const auto methods = type.getMethods();
          const auto symbol = typeInfoSymbol(type, "interface");
          const auto interfaceDataType =
            converter.convertType(converter.lookupRuntimeType("interfaceData"));
          return createGlobal(
            builder,
            module,
            interfaceDataType,
            symbol,
            loc,
            [&](OpBuilder& builder)
            {
              // Create data for each interface method.
              SmallVector<mlir::LLVM::GlobalOp> interfaceMethodDataGlobalOps;
              interfaceMethodDataGlobalOps.reserve(methods.size());
              for (auto [name, _func] : methods)
              {
                const auto func = mlir::cast<FunctionType>(_func);
                const auto id = computeMethodHash(name, func.getInputs(), func.getResults());
                const auto interfaceMethodSymbol =
                  typeInfoSymbol(type, "_interface_method_" + name + "_" + std::to_string(id));
                const auto interfaceMethodDataType =
                  converter.convertType(converter.lookupRuntimeType("interfaceMethodData"));
                auto interfaceMethodDataGlobalOp = createGlobal(
                  builder,
                  module,
                  interfaceMethodDataType,
                  interfaceMethodSymbol,
                  loc,
                  [&](OpBuilder& builder)
                  {
                    // Create the interface method data value.
                    Value dataValue =
                      mlir::LLVM::ZeroOp::create(builder, loc, interfaceMethodDataType);

                    // Insert the method hash id value.
                    Value methodIdValue = mlir::LLVM::ConstantOp::create(builder, 
                      loc, builder.getI32Type(), static_cast<uint64_t>(id));
                    dataValue =
                      mlir::LLVM::InsertValueOp::create(builder, loc, dataValue, methodIdValue, ArrayRef<int64_t>{0});

                    // Insert the method signature data.
                    auto methodSignatureDataGlobalOp = createTypeInfo(builder, module, loc, func, converter);
                    Value methodSignatureDataValue =
                      mlir::LLVM::AddressOfOp::create(builder, loc, methodSignatureDataGlobalOp);
                    dataValue = mlir::LLVM::InsertValueOp::create(builder,
                      loc, dataValue, methodSignatureDataValue, ArrayRef<int64_t>{1});

                    // Yield the interface method data value.
                    mlir::LLVM::ReturnOp::create(builder, loc, dataValue);
                  });
                interfaceMethodDataGlobalOps.push_back(interfaceMethodDataGlobalOp);
              }

              // Create the interface data value.
              Value dataValue = mlir::LLVM::ZeroOp::create(builder, loc, interfaceDataType);

              // Insert the interface methods data slice.
              Value interfaceMethodsDataSliceValue = createSliceValue(
                builder,
                module,
                converter,
                symbol + "_interface_methods",
                ptrType,
                interfaceMethodDataGlobalOps.size(),
                [&](OpBuilder& builder)
                {
                  SmallVector<Value> values;
                  values.reserve(interfaceMethodDataGlobalOps.size());
                  for (auto interfaceMethodDataGlobalOp : interfaceMethodDataGlobalOps)
                  {
                    // Get the address of the interface method data.
                    Value interfaceMethodDataValue =
                      mlir::LLVM::AddressOfOp::create(builder, loc, interfaceMethodDataGlobalOp);
                    values.push_back(interfaceMethodDataValue);
                  }
                  return values;
                },
                loc);
              dataValue = mlir::LLVM::InsertValueOp::create(builder,
                loc, dataValue, interfaceMethodsDataSliceValue, ArrayRef<int64_t>{0});

              // Yield the interface data value.
              mlir::LLVM::ReturnOp::create(builder, loc, dataValue);
            });
        })
      .Case<NamedType>(
        [&](NamedType type)
        {
          typeName = type.getName().getValue();
          const auto namedTypeDataSymbol = typeInfoSymbol(T, typeName.str());
          const auto underlyingType = type.getUnderlying();
          const auto methodSymbols = type.getMethods();

          // Create the type data for the named type's underlying type.
          auto underlyingTypeDataGlobalOp = createTypeInfo(builder, module, loc, underlyingType, converter);

          // Create the method data slice if there is metadata about them stored in the extra data
          // map.
          SmallVector<mlir::LLVM::GlobalOp> funcDataGlobalOps;
          const auto funcDataType = converter.convertType(converter.lookupRuntimeType("funcData"));

          // Pre-build a map from base symbol to instance functions for efficient
          // generic instance resolution (avoids up to 64 symbol lookups per method).
          llvm::StringMap<SmallVector<std::pair<std::string, mlir::FunctionOpInterface>>> instanceMap;
          module.walk([&](mlir::Operation* op) {
            if (auto funcIface = dyn_cast<mlir::FunctionOpInterface>(op))
            {
              StringRef name = funcIface.getName();
              auto dollarPos = name.find("$instance_");
              if (dollarPos != StringRef::npos)
              {
                StringRef base = name.substr(0, dollarPos);
                instanceMap[base].emplace_back(name.str(), funcIface);
              }
            }
          });

          // Create the function data globals.
          funcDataGlobalOps.reserve(methodSymbols.size());
          for (const auto& methodSymbol : methodSymbols)
          {
            const auto funcSymbol = cast<mlir::FlatSymbolRefAttr>(methodSymbol);

            mlir::go::FunctionType fnT;
            // Look up the function in the current module.
            auto funcOp = cast_or_null<mlir::FunctionOpInterface>(module.lookupSymbol(funcSymbol));

            // For generic type instances, the method symbol references the uninstantiated
            // generic name, but the compiled function has an $instance_N suffix.
            // Search for the matching instance by checking receiver type.
            std::string resolvedFuncSymbol = funcSymbol.getValue().str();
            if (!funcOp)
            {
              auto it = instanceMap.find(funcSymbol.getValue());
              if (it != instanceMap.end())
              {
                for (auto& [instanceSymbol, candidateOp] : it->second)
                {
                  // Verify this instance's receiver type matches the current named type.
                  if (auto attr =
                        candidateOp->getAttrOfType<mlir::TypeAttr>("originalType"))
                  {
                    if (auto candidateFnT =
                          dyn_cast<FunctionType>(attr.getValue()))
                    {
                      if (candidateFnT.hasReceiver())
                      {
                        auto recvType = candidateFnT.getReceiver();
                        if (auto ptrT = dyn_cast<PointerType>(recvType))
                          if (ptrT.getElementType().has_value())
                            recvType = *ptrT.getElementType();
                        if (recvType == type)
                        {
                          funcOp = candidateOp;
                          resolvedFuncSymbol = instanceSymbol;
                          break;
                        }
                      }
                    }
                  }
                }
              }
            }

            // Note: It is possible for some method functions to be dropped if there is no usage of
            // them. Do not generate information for these.
            if (!funcOp)
            {
              continue;
            }

            if (auto originalTypeAttr = funcOp->getAttrOfType<mlir::TypeAttr>("originalType"))
            {
              fnT = mlir::dyn_cast<mlir::go::FunctionType>(originalTypeAttr.getValue());
            }

            assert(fnT && "function type is unknown");

            // Compute the hash id for the type method.
            auto methodName = funcSymbol.getValue();
            methodName = methodName.substr(methodName.find_last_of(".") + 1);
            const auto methodHashId =
              computeMethodHash(methodName, fnT.getInputs(), fnT.getResults());

            // Create the type info for this function's signature.
            auto signatureTypeDataGlobalOp = createSignatureDataGlobal(builder, module, loc, fnT, converter);

            // Determine the symbol to use for the function pointer in the method table.
            // For value receiver methods, generate a wrapper thunk that loads the receiver
            // from a pointer before calling the actual method. This is needed because
            // interface dispatch always passes the receiver as a pointer (unsafe.Pointer),
            // but value receiver methods expect the receiver passed by value.
            std::string funcPtrSymbolStr = resolvedFuncSymbol;

            const bool hasValueReceiver =
              fnT.hasReceiver() && !mlir::go::isa<mlir::go::PointerType>(fnT.getReceiver());

            if (hasValueReceiver)
            {
              const std::string wrapperName = "__iface_thunk." + resolvedFuncSymbol;

              // Only create the wrapper if it doesn't already exist.
              if (!module.lookupSymbol(wrapperName))
              {
                // Get the LLVM function type of the actual method.
                mlir::LLVM::LLVMFunctionType llvmFuncType;
                if (auto llvmFunc = dyn_cast<mlir::LLVM::LLVMFuncOp>(funcOp.getOperation()))
                {
                  llvmFuncType = llvmFunc.getFunctionType();
                }
                else if (auto stdFunc = dyn_cast<mlir::func::FuncOp>(funcOp.getOperation()))
                {
                  mlir::TypeConverter::SignatureConversion sigConv(stdFunc.getNumArguments());
                  llvmFuncType = cast<mlir::LLVM::LLVMFunctionType>(
                    converter.convertFunctionSignature(
                      stdFunc.getFunctionType(), false, false, sigConv));
                }

                if (llvmFuncType && llvmFuncType.getNumParams() > 0)
                {
                  // The first parameter is the receiver type.
                  const auto receiverLLVMType = llvmFuncType.getParams()[0];

                  // Build wrapper param types: (ptr, param1, param2, ...)
                  SmallVector<mlir::Type> wrapperParams;
                  wrapperParams.push_back(ptrType);
                  for (size_t i = 1; i < llvmFuncType.getNumParams(); i++)
                    wrapperParams.push_back(llvmFuncType.getParams()[i]);

                  auto wrapperFnType = mlir::LLVM::LLVMFunctionType::get(
                    llvmFuncType.getReturnType(), wrapperParams);

                  // Create the wrapper function.
                  mlir::OpBuilder::InsertionGuard wrapperGuard(builder);
                  builder.setInsertionPointToEnd(module.getBody());

                  auto wrapperOp = mlir::LLVM::LLVMFuncOp::create(
                    builder, loc, wrapperName, wrapperFnType);
                  wrapperOp.setLinkage(mlir::LLVM::Linkage::Private);

                  auto* entry = wrapperOp.addEntryBlock(builder);
                  builder.setInsertionPointToStart(entry);

                  // Load the receiver value from the pointer.
                  Value recvPtr = entry->getArgument(0);
                  Value loadedRecv = mlir::LLVM::LoadOp::create(
                    builder, loc, receiverLLVMType, recvPtr);

                  // Build call args: [loaded_receiver, arg1, arg2, ...]
                  SmallVector<Value> callArgs;
                  callArgs.push_back(loadedRecv);
                  for (size_t i = 1; i < entry->getNumArguments(); i++)
                    callArgs.push_back(entry->getArgument(i));

                  // Call the actual method.
                  auto callOp = mlir::LLVM::CallOp::create(
                    builder, loc, llvmFuncType,
                    FlatSymbolRefAttr::get(builder.getContext(), resolvedFuncSymbol),
                    callArgs);
                  callOp.getProperties().operandSegmentSizes =
                    { { static_cast<int32_t>(callArgs.size()), 0 } };
                  callOp.getProperties().op_bundle_sizes =
                    builder.getDenseI32ArrayAttr({});

                  // Return results.
                  if (mlir::isa<mlir::LLVM::LLVMVoidType>(llvmFuncType.getReturnType()))
                  {
                    mlir::LLVM::ReturnOp::create(builder, loc, ValueRange{});
                  }
                  else
                  {
                    mlir::LLVM::ReturnOp::create(builder, loc, callOp->getResults());
                  }

                  funcPtrSymbolStr = wrapperName;
                }
              }
              else
              {
                // Wrapper already exists, just use its name.
                funcPtrSymbolStr = wrapperName;
              }
            }

            // Create the function data.
            const auto funcDataSymbol =
              typeInfoSymbol(T, typeName.str() + "_method_" + methodName.str());
            auto funcDataGlobalOp = createGlobal(
              builder,
              module,
              funcDataType,
              funcDataSymbol,
              loc,
              [&](OpBuilder& builder)
              {
                Value funcDataValue = mlir::LLVM::ZeroOp::create(builder, loc, funcDataType);

                // Insert the method id value.
                Value methodIdValue =
                  mlir::LLVM::ConstantOp::create(builder, loc, builder.getI32Type(), methodHashId);
                funcDataValue =
                  mlir::LLVM::InsertValueOp::create(builder, loc, funcDataValue, methodIdValue, ArrayRef<int64_t>{0});

                // Insert the address to the function (wrapper thunk or direct method).
                Value funcPtrValue =
                  mlir::LLVM::AddressOfOp::create(builder, loc, ptrType,
                    FlatSymbolRefAttr::get(builder.getContext(), funcPtrSymbolStr));
                funcDataValue =
                  mlir::LLVM::InsertValueOp::create(builder, loc, funcDataValue, funcPtrValue, ArrayRef<int64_t>{1});

                // Insert the address to the signature type data.
                Value signatureTypeDataValue =
                  mlir::LLVM::AddressOfOp::create(builder, loc, signatureTypeDataGlobalOp);
                funcDataValue = mlir::LLVM::InsertValueOp::create(builder,
                  loc, funcDataValue, signatureTypeDataValue, ArrayRef<int64_t>{2});

                // Yield the function data value.
                mlir::LLVM::ReturnOp::create(builder, loc, funcDataValue);
              });
            funcDataGlobalOps.push_back(funcDataGlobalOp);
          }

          const auto namedTypeDataType =
            converter.convertType(converter.lookupRuntimeType("namedTypeData"));
          return createGlobal(
            builder,
            module,
            namedTypeDataType,
            namedTypeDataSymbol,
            loc,
            [&](OpBuilder& builder)
            {
              // Create the data value.
              Value dataValue = mlir::LLVM::ZeroOp::create(builder, loc, namedTypeDataType);

              // Insert the underlying type data.
              Value underlyingTypeDataValue =
                mlir::LLVM::AddressOfOp::create(builder, loc, underlyingTypeDataGlobalOp);
              dataValue = mlir::LLVM::InsertValueOp::create(builder,
                loc, dataValue, underlyingTypeDataValue, ArrayRef<int64_t>{0});

              if (!funcDataGlobalOps.empty())
              {
                // Create the methods data slice value.
                Value nameTypeMethodsValue = createSliceValue(
                  builder,
                  module,
                  converter,
                  namedTypeDataSymbol,
                  ptrType,
                  funcDataGlobalOps.size(),
                  [&](OpBuilder& builder)
                  {
                    SmallVector<Value> values;
                    for (const auto& funcDataGlobalOp : funcDataGlobalOps)
                    {
                      Value funcDataValue =
                        mlir::LLVM::AddressOfOp::create(builder, loc, funcDataGlobalOp);
                      values.push_back(funcDataValue);
                    }
                    return values;
                  },
                  loc);

                // Insert the methods data slice value.
                dataValue = mlir::LLVM::InsertValueOp::create(builder,
                  loc, dataValue, nameTypeMethodsValue, ArrayRef<int64_t>{1});
              }

              mlir::LLVM::ReturnOp::create(builder, loc, dataValue);
            });
        })
      .Case<MapType>(
        [&](MapType type)
        {
          // Create the key type data.
          auto keyTypeDataGlobalOp = createTypeInfo(builder, module, loc, type.getKeyType(), converter);

          // Create the element type data.
          auto elementTypeDataGlobalOp = createTypeInfo(builder, module, loc, type.getValueType(), converter);

          // Create the map type data.
          const auto symbol = typeInfoSymbol(type, "map");
          const auto mapTypeDataType =
            converter.convertType(converter.lookupRuntimeType("mapTypeData"));
          return createGlobal(
            builder,
            module,
            mapTypeDataType,
            symbol,
            loc,
            [&](OpBuilder& builder)
            {
              // Create the map type date value.
              Value dataValue = mlir::LLVM::ZeroOp::create(builder, loc, mapTypeDataType);

              // Insert the key type data pointer value.
              Value keyTypeDataValue =
                mlir::LLVM::AddressOfOp::create(builder, loc, keyTypeDataGlobalOp);
              dataValue =
                mlir::LLVM::InsertValueOp::create(builder, loc, dataValue, keyTypeDataValue, ArrayRef<int64_t>{0});

              // Insert the element type data pointer value.
              Value elementTypeDataValue =
                mlir::LLVM::AddressOfOp::create(builder, loc, elementTypeDataGlobalOp);
              dataValue =
                mlir::LLVM::InsertValueOp::create(builder, loc, dataValue, elementTypeDataValue, ArrayRef<int64_t>{1});

              // Yield the map data value.
              mlir::LLVM::ReturnOp::create(builder, loc, dataValue);
            });
        })
      .Case<GoStructType>(
        [&](GoStructType type)
        {
          const auto fields = type.getFields();
          const auto symbol = typeInfoSymbol(type, "struct");
          const auto structTypeDataType =
            converter.convertType(converter.lookupRuntimeType("structTypeData"));
          const auto structFieldDataType =
            converter.convertType(converter.lookupRuntimeType("structFieldData"));

          // Pre-create type info globals for each field's type (outside the
          // initializer callbacks to avoid ordering issues).
          const auto dataLayout = mlir::DataLayout(module);
          SmallVector<mlir::LLVM::GlobalOp> fieldTypeInfoGlobals;
          SmallVector<uint64_t> fieldOffsets;
          fieldTypeInfoGlobals.reserve(fields.size());
          fieldOffsets.reserve(fields.size());
          for (size_t i = 0; i < fields.size(); i++)
          {
            auto [name, fieldType, tags] = fields[i];
            fieldTypeInfoGlobals.push_back(
              createTypeInfo(builder, module, loc, fieldType, converter));
            fieldOffsets.push_back(type.getFieldOffset(dataLayout, i));
          }

          return createGlobal(
            builder,
            module,
            structTypeDataType,
            symbol,
            loc,
            [&](OpBuilder& builder)
            {
              // Build the _structTypeData value.
              Value dataValue = mlir::LLVM::ZeroOp::create(builder, loc, structTypeDataType);

              if (!fields.empty())
              {
                // Create slice of _structFieldData values.
                Value fieldsSlice = createSliceValue(
                  builder,
                  module,
                  converter,
                  symbol + "_fields",
                  structFieldDataType,
                  fields.size(),
                  [&](OpBuilder& builder)
                  {
                    SmallVector<Value> values;
                    values.reserve(fields.size());
                    for (size_t i = 0; i < fields.size(); i++)
                    {
                      auto [name, fieldType, tags] = fields[i];

                      // Build a _structFieldData struct value inline.
                      Value fieldValue =
                        mlir::LLVM::ZeroOp::create(builder, loc, structFieldDataType);

                      // Insert dataType pointer (index 0).
                      Value typePtr =
                        mlir::LLVM::AddressOfOp::create(builder, loc, fieldTypeInfoGlobals[i]);
                      fieldValue = mlir::LLVM::InsertValueOp::create(
                        builder, loc, fieldValue, typePtr, ArrayRef<int64_t>{0});

                      // Insert tag string (index 1).
                      if (tags && !tags.empty())
                      {
                        Value tagValue = createGoStringValue(
                          builder, module, converter, tags.getValue(), loc);
                        fieldValue = mlir::LLVM::InsertValueOp::create(
                          builder, loc, fieldValue, tagValue, ArrayRef<int64_t>{1});
                      }

                      // Insert offset uintptr (index 2).
                      Value offsetValue = mlir::LLVM::ConstantOp::create(
                        builder, loc, uintptrType, fieldOffsets[i]);
                      fieldValue = mlir::LLVM::InsertValueOp::create(
                        builder, loc, fieldValue, offsetValue, ArrayRef<int64_t>{2});

                      values.push_back(fieldValue);
                    }
                    return values;
                  },
                  loc);
                dataValue = mlir::LLVM::InsertValueOp::create(
                  builder, loc, dataValue, fieldsSlice, ArrayRef<int64_t>{0});
              }

              // Yield the struct type data value.
              mlir::LLVM::ReturnOp::create(builder, loc, dataValue);
            });
        })
      .Case<SliceType>([&](SliceType type)
                       { return createTypeInfo(builder, module, loc, type.getElementType(), converter); })
      .Case<PointerType>(
        [&](PointerType type)
        {
          if (type.getElementType())
          {
            return createTypeInfo(builder, module, loc, *type.getElementType(), converter);
          }
          return mlir::LLVM::GlobalOp();
        })
      .Default([&](Type) { return ::mlir::LLVM::GlobalOp(); });

  // Emit the type information in the module as a global.
  return createGlobal(
    builder,
    module,
    infoType,
    infoSymbol,
    loc,
    [&](OpBuilder& builder)
    {
      // Create the initializer value for the resulting global.
      Value typeValue = mlir::LLVM::ZeroOp::create(builder, loc, infoType);

      // Insert the type kind value.
      const GoTypeId kind = GetGoTypeId(baseType(T));
      Value kindValue =
        mlir::LLVM::ConstantOp::create(builder, loc, i8Type, static_cast<uint64_t>(kind));
      typeValue =
        mlir::LLVM::InsertValueOp::create(builder, loc, typeValue, kindValue, ArrayRef<int64_t>{constKindIndex});

      // Insert the type size value.
      const auto dataLayout = mlir::DataLayout(module);
      const auto typeSize = dataLayout.getTypeSize(T);
      Value sizeValue =
        mlir::LLVM::ConstantOp::create(builder, loc, uintptrType, static_cast<uint64_t>(typeSize));
      typeValue =
        mlir::LLVM::InsertValueOp::create(builder, loc, typeValue, sizeValue, ArrayRef<int64_t>{constSizeIndex});

      if (dataGlobalOp)
      {
        // Insert the address to the respective type data.
        Value dataValue = mlir::LLVM::AddressOfOp::create(builder, loc, dataGlobalOp);
        typeValue =
          mlir::LLVM::InsertValueOp::create(builder, loc, typeValue, dataValue, ArrayRef<int64_t>{constDataIndex});
      }

      if (!typeName.empty())
      {
        Value nameValue = createGoStringValue(builder, module, converter, typeName, loc);
        typeValue =
          mlir::LLVM::InsertValueOp::create(builder, loc, typeValue, nameValue, ArrayRef<int64_t>{constNameIndex});
      }

      // Yield the type data value.
      mlir::LLVM::ReturnOp::create(builder, loc, typeValue);
    });
}
} // namespace mlir::go
