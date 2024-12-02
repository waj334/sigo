#include <llvm/ADT/TypeSwitch.h>

#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Dialect/Ptr/IR/PtrOpsDialect.h.inc>
#include <mlir/Pass/Pass.h>

#include "Go/IR/GoOps.h"
#include "Go/Transforms/TypeConverter.h"

namespace mlir::go
{

struct CallPass : public mlir::PassWrapper<CallPass, mlir::OperationPass<mlir::ModuleOp>>
{
  llvm::SmallDenseMap<llvm::hash_code, std::string> m_thunkSymbols;
  llvm::SmallDenseMap<llvm::hash_code, std::string> m_wrapperSymbols;

  std::pair<mlir::FlatSymbolRefAttr, mlir::Value> createCallWrapper(
    mlir::OpBuilder& builder,
    mlir::ModuleOp module,
    const mlir::Location loc,
    mlir::Value callee,
    mlir::ValueRange args,
    mlir::StringAttr method = mlir::StringAttr())
  {
    const auto ptrType = builder.getType<mlir::go::PointerType>(std::nullopt);

    SmallVector<mlir::Value> calleeArgs;
    mlir::Value calleeValue;
    bool isInterface = false;
    mlir::Type interfaceType;

    mlir::TypeRange resultTypes;
    mlir::TypeSwitch<mlir::Type>(mlir::go::baseType(callee.getType()))
      .Case(
        [&](const InterfaceType& type)
        {
          isInterface = true;
          calleeValue = callee;
          interfaceType = type;

          const auto methodSignature =
            mlir::cast<mlir::go::FunctionType>(type.getMethods()[method.str()]);
          resultTypes = methodSignature.getResults();
        })
      .Case(
        [&](const PointerType& type)
        {
          calleeValue = callee;
          const auto calleeType = *type.getElementType();
          const auto calleeSignature =
            mlir::dyn_cast_or_null<mlir::go::FunctionType>(mlir::go::baseType(calleeType));
          resultTypes = calleeSignature.getResults();
        })
      .Case(
        [&](const GoStructType& type)
        {
          // TODO: Probably should represent this one by some closure type.
          calleeValue = builder.create<ExtractOp>(loc, ptrType, 0, callee);
          mlir::Value argsPtrValue = builder.create<ExtractOp>(loc, ptrType, 1, callee);

          // Prepend the previous arguments pointer value to the argument list.
          calleeArgs.push_back(argsPtrValue);
        })
      .Default([&](const mlir::Type&) { assert(false && "unhandled callee type"); });

    // Append the incoming call arguments.
    calleeArgs.append(args.begin(), args.end());

    // Collect all the callee arg types.
    llvm::hash_code argsHash = llvm::hash_value(isInterface);
    SmallVector<std::tuple<mlir::StringAttr, mlir::Type, mlir::StringAttr>> ctxStructTypes;
    ctxStructTypes.emplace_back(
      builder.getStringAttr(""), calleeValue.getType(), builder.getStringAttr(""));

    // Hash and add the argument value types.
    for (const auto& arg : calleeArgs)
    {
      // Hash the type's unique storage pointer value.
      argsHash = llvm::hash_value(arg.getType().getImpl());
      ctxStructTypes.emplace_back(
        builder.getStringAttr(""), arg.getType(), builder.getStringAttr(""));
    }

    // Create a struct type for these call arguments.
    const auto ctxStructType =
      mlir::go::GoStructType::getLiteral(builder.getContext(), ctxStructTypes);

    // Create the context struct value.
    mlir::Value ctxValue = builder.create<mlir::go::ZeroOp>(loc, ctxStructType);
    ctxValue = builder.create<mlir::go::InsertOp>(loc, ctxStructType, calleeValue, 0, ctxValue);
    for (size_t i = 0; i < calleeArgs.size(); i++)
    {
      ctxValue =
        builder.create<mlir::go::InsertOp>(loc, ctxStructType, calleeArgs[i], i + 1, ctxValue);
    }

    std::string symbol;
    if (const auto it = this->m_wrapperSymbols.find(argsHash); it == this->m_wrapperSymbols.end())
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(module.getBody());

      // Format the wrapper symbol name.
      symbol = "_call_wrapper_" + std::to_string(argsHash);

      // Create the signature for this function.
      auto signature = builder.getType<mlir::go::FunctionType>(
        mlir::SmallVector<mlir::Type>{ ptrType }, resultTypes);

      // Create a new call wrapper function.

      {
        auto funcOp = builder.create<FuncOp>(loc, symbol, signature);
        auto entryBlock = funcOp.addEntryBlock();

        mlir::OpBuilder::InsertionGuard guard2(builder);
        builder.setInsertionPointToStart(entryBlock);
        const mlir::Value closureValue = entryBlock->getArgument(0);

        // Unpack the call arguments.
        SmallVector<Value> callArgs(calleeArgs.size());
        for (int32_t i = 0; i < static_cast<int32_t>(callArgs.size()); ++i)
        {
          const auto argType = calleeArgs[i].getType();
          Value callArgPtr = builder.create<GetElementPointerOp>(
            loc,
            ptrType,
            closureValue,
            ctxStructType,
            ValueRange{},
            SmallVector<int32_t>{ 0, i + 1 });
          callArgs[i] = builder.create<LoadOp>(loc, argType, callArgPtr, UnitAttr(), UnitAttr());
        }

        mlir::ValueRange results;
        if (isInterface)
        {
          // Unpack the interface receiver value.
          Value interfaceValue =
            builder.create<LoadOp>(loc, interfaceType, closureValue, UnitAttr(), UnitAttr());

          // Call the function being wrapped.
          results =
            builder.create<InterfaceCallOp>(loc, resultTypes, method, interfaceValue, callArgs)
              .getResults();
        }
        else
        {
          // Unpack the callee function pointer.
          Value funcPtr =
            builder.create<LoadOp>(loc, ptrType, closureValue, UnitAttr(), UnitAttr());

          // Call the function being wrapped.
          results =
            builder.create<CallIndirectOp>(loc, resultTypes, funcPtr, callArgs).getResults();
        }

        // Create return operation.
        builder.create<mlir::go::ReturnOp>(loc, results);
      }

      // Cache this wrapper symbol.
      this->m_wrapperSymbols[argsHash] = symbol;
    }
    else
    {
      symbol = it->second;
    }
    return std::make_pair(builder.getAttr<mlir::FlatSymbolRefAttr>(symbol), ctxValue);
  }

  void runOnOperation() final
  {
    auto context = &this->getContext();
    auto module = getOperation();

    mlir::DataLayout dataLayout(module);
    mlir::LowerToLLVMOptions options(&getContext(), dataLayout);
    if (auto dataLayoutStr = dyn_cast<StringAttr>(module->getAttr("llvm.data_layout"));
        dataLayoutStr)
    {
      llvm::DataLayout llvmDataLayout(dataLayoutStr);
      options.dataLayout = llvmDataLayout;
    }

    auto typeConverter = mlir::go::LLVMTypeConverter(module, options);

    const auto ptrType = PointerType::get(context, std::nullopt);
    // const auto interfaceType = typeConverter.lookupRuntimeType("interface");
    const auto funcType = typeConverter.lookupRuntimeType("func");

    /*
    auto createGeneralCallThunk =
      [&](OpBuilder& builder, FunctionType fnT, Type argPackType) -> std::string
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(module.getBody());

      // Note: Since thunks can be reused for calls with matching signature, no fixed location for
      // the resulting operations can be known.
      const auto loc = UnknownLoc::get(context);

      // Hash the call argument types.
      llvm::hash_code argsHash{};
      for (const auto& argType : fnT.getInputs())
      {
        // Hash the type's unique storage pointer value.
        argsHash = llvm::hash_value(argType.getImpl());
      }

      // Integrate the call arguments hash into the thunk function symbol name.
      std::string funcSymbolName = "thunk_func_" + std::to_string(argsHash);

      // Look up the thunk symbol by hash value and create the function if it is not found in the
      // cache.
      if (const auto it = this->m_thunkSymbols.find(argsHash); it == this->m_thunkSymbols.end())
      {
        FunctionType signature = FunctionType::get(context, { ptrType }, fnT.getResults());

        // Create a function operation for the thunk.
        auto funcOp = builder.create<FuncOp>(loc, funcSymbolName, signature);
        auto entryBlock = funcOp.addEntryBlock();

        // Build the function body.
        {
          mlir::OpBuilder::InsertionGuard guard2(builder);
          builder.setInsertionPointToStart(entryBlock);
          Value argsPtr = entryBlock->getArgument(0);

          // Unpack the callee function pointer.
          Value funcPtr = builder.create<LoadOp>(loc, fnT, argsPtr, UnitAttr(), UnitAttr());

          // Unpack the call arguments.
          SmallVector<Value> callArgs(fnT.getNumInputs());
          for (int32_t i = 0; i < static_cast<int32_t>(callArgs.size()); ++i)
          {
            const auto argType = fnT.getInput(i);
            Value callArgPtr = builder.create<GetElementPointerOp>(
              loc, ptrType, argsPtr, argPackType, ValueRange{}, SmallVector<int32_t>{ 0, i + 1 });
            callArgs[i] = builder.create<LoadOp>(loc, argType, callArgPtr, UnitAttr(), UnitAttr());
          }

          // Call the function being wrapped.
          ValueRange results =
            builder.create<CallIndirectOp>(loc, fnT.getResults(), funcPtr, callArgs).getResults();

          // Create return operation.
          builder.create<func::ReturnOp>(loc, results);
        }

        // Cache the symbol.
        this->m_thunkSymbols[argsHash] = funcSymbolName;
      }
      return funcSymbolName;
    };

    auto createInterfaceCallThunk = [&](
                                      OpBuilder& builder,
                                      FunctionType fnT,
                                      Type interfaceType,
                                      StringRef callee,
                                      ValueRange args,
                                      Type argPackType) -> std::string
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(module.getBody());

      // Note: Since thunks can be reused for calls with matching signature, no fixed location for
      // the
      //       resulting operations can be known.
      const auto loc = UnknownLoc::get(context);

      // Hash the call argument types.
      // TODO: This can be optimized further by lowering directly to runtime calls so the runtime
      //       representation of the interface type can be used.
      llvm::hash_code argsHash{};
      SmallVector<Type> argTypes = { interfaceType };
      llvm::append_range(argTypes, args.getTypes());
      for (const auto& argType : argTypes)
      {
        // Hash the type's unique storage pointer value.
        argsHash = llvm::hash_value(argType.getImpl());
      }

      // Integrate the call arguments hash into the thunk function symbol name.
      std::string funcSymbolName = "thunk_iface_func_" + std::to_string(argsHash);

      // Look up the thunk symbol by hash value and create the function if it is not found in the
      // cache.
      if (const auto it = this->m_thunkSymbols.find(argsHash); it == this->m_thunkSymbols.end())
      {
        FunctionType signature = FunctionType::get(context, { ptrType }, fnT.getResults());

        // Create a function operation for the thunk.
        auto funcOp = builder.create<FuncOp>(loc, funcSymbolName, signature);
        auto entryBlock = funcOp.addEntryBlock();

        // Build the function body.
        {
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(entryBlock);
          Value argsPtr = entryBlock->getArgument(0);

          // Unpack the interface receiver value.
          Value interfaceValue =
            builder.create<LoadOp>(loc, interfaceType, argsPtr, UnitAttr(), UnitAttr());

          // Unpack the call arguments.
          SmallVector<Value> callArgs(fnT.getNumInputs());
          for (int32_t i = 0; i < static_cast<int32_t>(args.size()); ++i)
          {
            const auto argType = fnT.getInput(i);
            Value callArgPtr = builder.create<GetElementPointerOp>(
              loc, ptrType, argsPtr, argPackType, ValueRange{}, SmallVector<int32_t>{ 0, i + 1 });
            callArgs[i] = builder.create<LoadOp>(loc, argType, callArgPtr, UnitAttr(), UnitAttr());
          }

          // Call the function being wrapped.
          auto results =
            builder.create<InterfaceCallOp>(loc, TypeRange{}, callee, interfaceValue, callArgs)
              .getResults();

          // Create return operation.
          builder.create<func::ReturnOp>(loc, results);
        }

        // Cache the symbol.
        this->m_thunkSymbols[argsHash] = funcSymbolName;
      }
      return funcSymbolName;
    };
*/

    // Walk all defer calls and make sure all return paths in the parent function run defers before
    // exiting.
    mlir::DenseSet<mlir::Operation*> visitedFuncs;
    module.walk(
      [&](DeferOp deferOp)
      {
        const auto loc = deferOp.getLoc();
        std::optional<std::pair<mlir::FlatSymbolRefAttr, mlir::Value>> wrappedCallee;
        OpBuilder builder(deferOp);
        if (mlir::go::isa<mlir::go::GoStructType>(deferOp.getCallee().getType()))
        {
          if (deferOp.getCalleeOperands().size() > 0)
          {
            // Create a call wrapper for any previously wrapped call that specifies more arguments.
            wrappedCallee = createCallWrapper(
              builder,
              module,
              deferOp.getLoc(),
              deferOp.getCallee(),
              deferOp.getCalleeOperands(),
              deferOp.getMethodNameAttr());
          }
        }
        else
        {
          wrappedCallee = createCallWrapper(
            builder,
            module,
            deferOp.getLoc(),
            deferOp.getCallee(),
            deferOp.getCalleeOperands(),
            deferOp.getMethodNameAttr());
        }

        if (wrappedCallee.has_value())
        {
          const auto symbol = wrappedCallee.value().first;
          const auto args = wrappedCallee.value().second;

          // Get the call wrapper function by symbol.
          Value funcPtr = builder.create<AddressOfOp>(loc, ptrType, symbol);

          // Allocate memory to store the call args.
          Value argsPtr = builder.create<AllocaOp>(
            loc, ptrType, args.getType(), 1, builder.getUnitAttr(), StringAttr());
          builder.create<StoreOp>(loc, args, argsPtr, UnitAttr(), UnitAttr());

          // Create the func value.
          mlir::Value funcValue = builder.create<ZeroOp>(loc, funcType);
          funcValue = builder.create<InsertOp>(loc, funcType, funcPtr, 0, funcValue);
          funcValue = builder.create<InsertOp>(loc, funcType, argsPtr, 1, funcValue);

          // Replace the defer op call.
          const auto newDeferOp = builder.create<mlir::go::DeferOp>(
            loc, funcValue, mlir::StringAttr(), mlir::ValueRange());
          deferOp->erase();
          deferOp = newDeferOp;
        }

        // Handle RunDefersOp insertion for the parent function.
        auto parentFunction = deferOp->getParentOfType<mlir::go::FuncOp>();
        if (visitedFuncs.contains(parentFunction))
        {
          return mlir::WalkResult::skip();
        }

        parentFunction.walk(
          [&](mlir::go::ReturnOp returnOp)
          {
            OpBuilder builder(returnOp);
            builder.create<mlir::go::RunDefersOp>(returnOp.getLoc());
          });

        return mlir::WalkResult::advance();
      });

    module.walk(
      [&](GoOp goOp)
      {
        const auto loc = goOp.getLoc();
        std::optional<std::pair<mlir::FlatSymbolRefAttr, mlir::Value>> wrappedCallee;
        OpBuilder builder(goOp);
        if (mlir::go::isa<mlir::go::GoStructType>(goOp.getCallee().getType()))
        {
          if (goOp.getCalleeOperands().size() > 0)
          {
            // Create a call wrapper for any previously wrapped call that specifies more arguments.
            wrappedCallee = createCallWrapper(
              builder,
              module,
              goOp.getLoc(),
              goOp.getCallee(),
              goOp.getCalleeOperands(),
              goOp.getMethodNameAttr());
          }
        }
        else
        {
          wrappedCallee = createCallWrapper(
            builder,
            module,
            goOp.getLoc(),
            goOp.getCallee(),
            goOp.getCalleeOperands(),
            goOp.getMethodNameAttr());
        }

        if (wrappedCallee.has_value())
        {
          const auto symbol = wrappedCallee.value().first;
          const auto args = wrappedCallee.value().second;

          // Get the call wrapper function by symbol.
          Value funcPtr = builder.create<AddressOfOp>(loc, ptrType, symbol);

          // Allocate memory to store the call args.
          Value argsPtr = builder.create<AllocaOp>(
            loc, ptrType, args.getType(), 1, builder.getUnitAttr(), StringAttr());
          builder.create<StoreOp>(loc, args, argsPtr, UnitAttr(), UnitAttr());

          // Create the func value.
          mlir::Value funcValue = builder.create<ZeroOp>(loc, funcType);
          funcValue = builder.create<InsertOp>(loc, funcType, funcPtr, 0, funcValue);
          funcValue = builder.create<InsertOp>(loc, funcType, argsPtr, 1, funcValue);

          // Replace the defer op call.
          builder.create<mlir::go::GoOp>(loc, funcValue, mlir::StringAttr(), mlir::ValueRange());
          goOp->erase();
        }
      });
  }
};

std::unique_ptr<mlir::Pass> createCallPass()
{
  return std::make_unique<CallPass>();
}

} // namespace mlir::go
