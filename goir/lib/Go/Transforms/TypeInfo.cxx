#include "Go/Transforms/TypeInfo.h"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/CommandLine.h>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>

#include "Go/Util.h"

namespace mlir::go
{
static llvm::SmallDenseMap<mlir::Type, uint64_t> s_generatedTypeInfoMap =
  llvm::SmallDenseMap<mlir::Type, uint64_t>();
static uint64_t s_typeInfoCounter = 0;
static std::mutex s_typeInfoMutex = std::mutex();

uint64_t getTypeId(const mlir::Type& type)
{
  std::lock_guard<std::mutex> guard(s_typeInfoMutex);
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
  return builder.create<::mlir::LLVM::GlobalOp>(
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
    cstrValue = builder.create<mlir::LLVM::AddressOfOp>(loc, cstrGlobalOp);
  }
  else
  {
    // Create a new global C char string to hold the string data.
    cstrValue =
      mlir::LLVM::createGlobalString(loc, builder, cstrName, value, mlir::LLVM::Linkage::External);
  }

  // Create the Go string value.
  Value gostrValue = builder.create<mlir::LLVM::ZeroOp>(loc, gostrType);

  // Insert the C string address into the Go string struct value.
  gostrValue = builder.create<mlir::LLVM::InsertValueOp>(loc, gostrValue, cstrValue, 0);

  // Insert the string length value into the Go string struct value return the resulting Go string
  // value.
  Value lengthValue =
    builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), value.size());
  return builder.create<mlir::LLVM::InsertValueOp>(loc, gostrValue, lengthValue, 1);
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
      Value arrValue = builder.create<mlir::LLVM::ZeroOp>(loc, arrayType);
      for (size_t i = 0; i < length; i++)
      {
        arrValue = builder.create<mlir::LLVM::InsertValueOp>(loc, arrValue, values[i], i);
      }
      builder.create<mlir::LLVM::ReturnOp>(loc, arrValue);
    });

  // Create the Go slice value.
  Value goSliceValue = builder.create<mlir::LLVM::ZeroOp>(loc, goSliceType);

  // Insert the address of the backing array for the slice.
  Value arrValue = builder.create<mlir::LLVM::AddressOfOp>(loc, arrGlobalOp);
  goSliceValue = builder.create<mlir::LLVM::InsertValueOp>(loc, goSliceValue, arrValue, 0);

  // Insert the length and capacity values.
  Value lengthValue = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), length);
  goSliceValue = builder.create<mlir::LLVM::InsertValueOp>(loc, goSliceValue, lengthValue, 1);
  goSliceValue = builder.create<mlir::LLVM::InsertValueOp>(loc, goSliceValue, lengthValue, 2);
  return goSliceValue;
}

mlir::go::LLVMTypeConverter getLLVMTypeConverter(mlir::ModuleOp module)
{
  mlir::DataLayout dataLayout(module);
  mlir::LowerToLLVMOptions options(module.getContext(), dataLayout);
  if (auto dataLayoutStr = dyn_cast<StringAttr>(module->getAttr("llvm.data_layout")); dataLayoutStr)
  {
    llvm::DataLayout llvmDataLayout(dataLayoutStr);
    options.dataLayout = llvmDataLayout;
  }
  return mlir::go::LLVMTypeConverter(module, options);
}

llvm::hash_code computeLLVMFunctionHash(
  const StringRef name,
  const mlir::LLVM::LLVMFunctionType func,
  bool isInterface)
{
  llvm::hash_code result = llvm::hash_value(name);
  size_t offset = 0;
  if (!isInterface)
  {
    // Skip the receiver for named-type methods.
    offset = 1;
  }

  // Hash the input types starting at the offset.
  for (size_t i = offset; i < func.getNumParams(); i++)
  {
    std::string str;
    llvm::raw_string_ostream(str) << func.getParams()[i];
    result = llvm::hash_combine(result, str);
  }

  // Hash the result types
  for (auto t : func.getReturnTypes())
  {
    std::string str;
    llvm::raw_string_ostream(str) << t;
    result = llvm::hash_combine(result, str);
  }
  return result;
}

mlir::LLVM::GlobalOp createSignatureDataGlobal(
  mlir::OpBuilder& builder,
  mlir::ModuleOp module,
  const mlir::Location& loc,
  const mlir::go::FunctionType type)
{
  auto converter = getLLVMTypeConverter(module);
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
      Value dataValue = builder.create<mlir::LLVM::ZeroOp>(loc, dataType);

      // Insert the receiver type data if present.
      if (const auto receiverType = type.getReceiver())
      {
        auto receiverTypeDataGlobalOp = createTypeInfo(builder, module, loc, receiverType);
        Value receiverTypeDataValue =
          builder.create<mlir::LLVM::AddressOfOp>(loc, receiverTypeDataGlobalOp);
        dataValue =
          builder.create<mlir::LLVM::InsertValueOp>(loc, dataValue, receiverTypeDataValue, 0);
      }

      const uint64_t numInputs = type.getNumInputs();
      auto inputGeneratorFn = [&](OpBuilder& builder)
      {
        SmallVector<Value> inputTypeDataValues;
        inputTypeDataValues.reserve(numInputs);
        for (auto i = 0; i < type.getNumInputs(); i++)
        {
          // Create the type info for the input type.
          auto inputTypeInfoGlobalOp = createTypeInfo(builder, module, loc, type.getInput(i));

          // Get the address of the input type info.
          Value inputTypeInfoValue =
            builder.create<mlir::LLVM::AddressOfOp>(loc, inputTypeInfoGlobalOp);
          inputTypeDataValues.push_back(inputTypeInfoValue);
        }
        return inputTypeDataValues;
      };

      // Insert the slice value for the inputs type data.
      Value inputsSliceValue = createSliceValue(
        builder, module, converter, symbol + "_inputs", ptrType, numInputs, inputGeneratorFn, loc);
      dataValue = builder.create<mlir::LLVM::InsertValueOp>(loc, dataValue, inputsSliceValue, 1);

      auto resultGeneratorFn = [&](OpBuilder& builder)
      {
        SmallVector<Value> resultTypeDataValues;
        resultTypeDataValues.reserve(type.getNumResults());
        for (size_t i = 0; i < type.getNumResults(); i++)
        {
          // Create the type info for the result type.
          auto resultTypeInfoGlobalOp = createTypeInfo(builder, module, loc, type.getResult(i));

          // Get the address of the result type info.
          Value resultTypeInfoValue =
            builder.create<mlir::LLVM::AddressOfOp>(loc, resultTypeInfoGlobalOp);
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
      dataValue = builder.create<mlir::LLVM::InsertValueOp>(loc, dataValue, resultsSliceValue, 2);

      // Yield the function type data.
      builder.create<mlir::LLVM::ReturnOp>(loc, dataValue);
    });
}

mlir::LLVM::GlobalOp createTypeInfo(
  mlir::OpBuilder& builder,
  mlir::ModuleOp module,
  const mlir::Location& loc,
  const mlir::Type T)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());

  auto converter = getLLVMTypeConverter(module);
  const auto i8Type = builder.getI8Type();
  const auto i16Type = builder.getI16Type();
  const auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  const auto infoType = converter.convertType(converter.lookupRuntimeType("type"));

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
            createTypeInfo(builder, module, loc, type.getElementType());

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
              Value dataValue = builder.create<mlir::LLVM::ZeroOp>(loc, arrayTypeDataType);

              // Insert the length value.
              Value lengthValue =
                builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), type.getLength());
              dataValue = builder.create<mlir::LLVM::InsertValueOp>(loc, dataValue, lengthValue, 0);

              // Insert the element type data pointer value.
              Value elementTypeDataValue =
                builder.create<mlir::LLVM::AddressOfOp>(loc, elementTypeDataGlobalOp);
              dataValue =
                builder.create<mlir::LLVM::InsertValueOp>(loc, dataValue, elementTypeDataValue, 1);

              // Yield the array type data value.
              builder.create<mlir::LLVM::ReturnOp>(loc, dataValue);
            });
        })
      .Case<ChanType>(
        [&](ChanType type)
        {
          // Create the element type data.
          auto elementTypeDataGlobalOp =
            createTypeInfo(builder, module, loc, type.getElementType());

          // Create the chan type data.
          const auto symbol = typeInfoSymbol(type, "chan");
          const auto chanTypeDataType =
            converter.convertType(converter.lookupRuntimeType("chanTypeData"));
          return createGlobal(
            builder,
            module,
            chanTypeDataType,
            symbol,
            loc,
            [&](OpBuilder& builder)
            {
              // Create the chan type data value.
              Value dataValue = builder.create<mlir::LLVM::ZeroOp>(loc, chanTypeDataType);

              // Insert the element type data value.
              Value elementTypeDataValue =
                builder.create<mlir::LLVM::AddressOfOp>(loc, elementTypeDataGlobalOp);
              dataValue =
                builder.create<mlir::LLVM::InsertValueOp>(loc, dataValue, elementTypeDataValue, 0);

              // Insert the direction value.
              Value directionVal = builder.create<mlir::LLVM::ConstantOp>(
                loc, builder.getI8Type(), static_cast<uint64_t>(type.getDirection()));
              dataValue =
                builder.create<mlir::LLVM::InsertValueOp>(loc, dataValue, directionVal, 1);

              // Yield the chan type data value.
              builder.create<mlir::LLVM::ReturnOp>(loc, dataValue);
            });
        })
      .Case([&](mlir::go::FunctionType type)
            { return createSignatureDataGlobal(builder, module, loc, type); })
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
                const auto id = computeMethodHash(name, func, true);
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
                      builder.create<mlir::LLVM::ZeroOp>(loc, interfaceMethodDataType);

                    // Insert the method hash id value.
                    Value methodIdValue = builder.create<mlir::LLVM::ConstantOp>(
                      loc, builder.getI32Type(), static_cast<uint64_t>(id));
                    dataValue =
                      builder.create<mlir::LLVM::InsertValueOp>(loc, dataValue, methodIdValue, 0);

                    // Insert the method signature data.
                    auto methodSignatureDataGlobalOp = createTypeInfo(builder, module, loc, func);
                    Value methodSignatureDataValue =
                      builder.create<mlir::LLVM::AddressOfOp>(loc, methodSignatureDataGlobalOp);
                    dataValue = builder.create<mlir::LLVM::InsertValueOp>(
                      loc, dataValue, methodSignatureDataValue, 1);

                    // Yield the interface method data value.
                    builder.create<mlir::LLVM::ReturnOp>(loc, dataValue);
                  });
                interfaceMethodDataGlobalOps.push_back(interfaceMethodDataGlobalOp);
              }

              // Create the interface data value.
              Value dataValue = builder.create<mlir::LLVM::ZeroOp>(loc, interfaceDataType);

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
                      builder.create<mlir::LLVM::AddressOfOp>(loc, interfaceMethodDataGlobalOp);
                    values.push_back(interfaceMethodDataValue);
                  }
                  return values;
                },
                loc);
              dataValue = builder.create<mlir::LLVM::InsertValueOp>(
                loc, dataValue, interfaceMethodsDataSliceValue, 0);

              // Yield the interface data value.
              builder.create<mlir::LLVM::ReturnOp>(loc, dataValue);
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
          auto underlyingTypeDataGlobalOp = createTypeInfo(builder, module, loc, underlyingType);

          // Create the method data slice if there is metadata about them stored in the extra data
          // map.
          SmallVector<mlir::LLVM::GlobalOp> funcDataGlobalOps;
          const auto funcDataType = converter.convertType(converter.lookupRuntimeType("funcData"));

          // Create the function data globals.
          funcDataGlobalOps.reserve(methodSymbols.size());
          for (const auto& methodSymbol : methodSymbols)
          {
            const auto funcSymbol = cast<mlir::FlatSymbolRefAttr>(methodSymbol);

            mlir::go::FunctionType fnT;
            // Look up the function in the current module.
            auto funcOp = cast_or_null<mlir::FunctionOpInterface>(module.lookupSymbol(funcSymbol));

            // Note: It is possible for some method functions to be dropped if there is no usage of
            // them. Do not generate information for these.
            if (!funcOp)
            {
              continue;
            }

            if (
              mlir::TypeAttr originalTypeAttr =
                funcOp->getAttrOfType<mlir::TypeAttr>("originalType"))
            {
              fnT = mlir::dyn_cast<mlir::go::FunctionType>(originalTypeAttr.getValue());
            }

            assert(fnT && "function type is unknown");

            // Compute the hash id for the type method.
            auto methodName = funcSymbol.getValue();
            methodName = methodName.substr(methodName.find_last_of(".") + 1);
            const auto methodHashId = computeMethodHash(methodName, fnT, false);

            // Create the type info for this function's signature.
            auto signatureTypeDataGlobalOp = createSignatureDataGlobal(builder, module, loc, fnT);

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
                Value funcDataValue = builder.create<mlir::LLVM::ZeroOp>(loc, funcDataType);

                // Insert the method id value.
                Value methodIdValue =
                  builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), methodHashId);
                funcDataValue =
                  builder.create<mlir::LLVM::InsertValueOp>(loc, funcDataValue, methodIdValue, 0);

                // Insert the address to the function.
                Value funcPtrValue =
                  builder.create<mlir::LLVM::AddressOfOp>(loc, ptrType, funcSymbol);
                funcDataValue =
                  builder.create<mlir::LLVM::InsertValueOp>(loc, funcDataValue, funcPtrValue, 1);

                // Insert the address to the signature type data.
                Value signatureTypeDataValue =
                  builder.create<mlir::LLVM::AddressOfOp>(loc, signatureTypeDataGlobalOp);
                funcDataValue = builder.create<mlir::LLVM::InsertValueOp>(
                  loc, funcDataValue, signatureTypeDataValue, 2);

                // Yield the function data value.
                builder.create<mlir::LLVM::ReturnOp>(loc, funcDataValue);
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
              Value dataValue = builder.create<mlir::LLVM::ZeroOp>(loc, namedTypeDataType);

              // Insert the underlying type data.
              Value underlyingTypeDataValue =
                builder.create<mlir::LLVM::AddressOfOp>(loc, underlyingTypeDataGlobalOp);
              dataValue = builder.create<mlir::LLVM::InsertValueOp>(
                loc, dataValue, underlyingTypeDataValue, 0);

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
                        builder.create<mlir::LLVM::AddressOfOp>(loc, funcDataGlobalOp);
                      values.push_back(funcDataValue);
                    }
                    return values;
                  },
                  loc);

                // Insert the methods data slice value.
                dataValue = builder.create<mlir::LLVM::InsertValueOp>(
                  loc, dataValue, nameTypeMethodsValue, 1);
              }

              builder.create<mlir::LLVM::ReturnOp>(loc, dataValue);
            });
        })
      .Case<MapType>(
        [&](MapType type)
        {
          // Create the key type data.
          auto keyTypeDataGlobalOp = createTypeInfo(builder, module, loc, type.getKeyType());

          // Create the element type data.
          auto elementTypeDataGlobalOp = createTypeInfo(builder, module, loc, type.getValueType());

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
              Value dataValue = builder.create<mlir::LLVM::ZeroOp>(loc, mapTypeDataType);

              // Insert the key type data pointer value.
              Value keyTypeDataValue =
                builder.create<mlir::LLVM::AddressOfOp>(loc, keyTypeDataGlobalOp);
              dataValue =
                builder.create<mlir::LLVM::InsertValueOp>(loc, dataValue, keyTypeDataValue, 0);

              // Insert the element type data pointer value.
              Value elementTypeDataValue =
                builder.create<mlir::LLVM::AddressOfOp>(loc, elementTypeDataGlobalOp);
              dataValue =
                builder.create<mlir::LLVM::InsertValueOp>(loc, dataValue, elementTypeDataValue, 1);

              // Yield the map data value.
              builder.create<mlir::LLVM::ReturnOp>(loc, dataValue);
            });
        })
      .Case<SliceType>([&](SliceType type)
                       { return createTypeInfo(builder, module, loc, type.getElementType()); })
      .Case<PointerType>(
        [&](PointerType type)
        {
          if (type.getElementType())
          {
            return createTypeInfo(builder, module, loc, *type.getElementType());
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
      Value typeValue = builder.create<mlir::LLVM::ZeroOp>(loc, infoType);

      // Insert the type kind value.
      const GoTypeId kind = GetGoTypeId(baseType(T));
      Value kindValue =
        builder.create<mlir::LLVM::ConstantOp>(loc, i8Type, static_cast<uint64_t>(kind));
      typeValue = builder.create<mlir::LLVM::InsertValueOp>(loc, typeValue, kindValue, 0);

      // Insert the type size value.
      const auto dataLayout = mlir::DataLayout(module);
      const auto typeSize = dataLayout.getTypeSize(T);
      Value sizeValue =
        builder.create<mlir::LLVM::ConstantOp>(loc, i16Type, static_cast<uint64_t>(typeSize));
      typeValue = builder.create<mlir::LLVM::InsertValueOp>(loc, typeValue, sizeValue, 1);

      if (dataGlobalOp)
      {
        // Insert the address to the respective type data.
        Value dataValue = builder.create<mlir::LLVM::AddressOfOp>(loc, dataGlobalOp);
        typeValue = builder.create<mlir::LLVM::InsertValueOp>(loc, typeValue, dataValue, 2);
      }

      if (!typeName.empty())
      {
        Value nameValue = createGoStringValue(builder, module, converter, typeName, loc);
        typeValue = builder.create<mlir::LLVM::InsertValueOp>(loc, typeValue, nameValue, 3);
      }

      // Yield the type data value.
      builder.create<mlir::LLVM::ReturnOp>(loc, typeValue);
    });
}
} // namespace mlir::go
