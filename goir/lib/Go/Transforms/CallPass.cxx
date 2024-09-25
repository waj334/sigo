#include <llvm/ADT/TypeSwitch.h>

#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Pass/Pass.h>

#include "Go/IR/GoOps.h"
#include "Go/Transforms/TypeConverter.h"

namespace mlir::go
{
struct CallPass : public mlir::PassWrapper<CallPass, mlir::OperationPass<mlir::ModuleOp>>
{
  llvm::SmallDenseMap<llvm::hash_code, std::string> m_thunkSymbols;

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

    auto createGeneralCallThunk =
      [&](OpBuilder& builder, FunctionType fnT, Type argPackType) -> std::string
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(module.getBody());

      // Note: Since thunks can be reused for calls with matching signature, no fixed location for
      // the
      //       resulting operations can be known.
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

    // Walk all `go` operations and generate thunks.
    module.walk(
      [&](GoOp op)
      {
        IRRewriter rewriter(op);

        const auto loc = op.getLoc();
        const auto calleeType = go::baseType(op.getCallee().getType());
        const auto callArgs = op.getCalleeOperands();
        const auto signature = mlir::cast<FunctionType>(op.getSignature());

        Value funcValue;

        TypeSwitch<Type>(calleeType)
          .Case(
            [&](const FunctionType T)
            {
              if (T.getNumInputs() > 0)
              {
                // Collect the expected argument pack struct member types.
                // NOTE: The pointer to the callee is the first argument.
                SmallVector<Type> argTypes = { ptrType };
                llvm::append_range(argTypes, op.getCalleeOperands().getTypes());
                const auto argPackType = GoStructType::getBasic(context, argTypes);

                // Find or create a thunk to wrap the callee so that it's call arguments can be
                // unpacked properly.
                const auto funcSymbol = createGeneralCallThunk(rewriter, signature, argPackType);

                // Get the thunk function by symbol.
                Value funcPtr = rewriter.create<AddressOfOp>(loc, ptrType, funcSymbol);

                // Allocate memory to store the call args.
                Value args =
                  rewriter.create<AllocaOp>(loc, ptrType, argPackType, 1, UnitAttr(), StringAttr());

                // Pack the pointer to the callee as the first call argument.
                Value calleeFuncPtr = rewriter.create<BitcastOp>(loc, ptrType, op.getCallee());
                rewriter.create<StoreOp>(loc, calleeFuncPtr, args, UnitAttr(), UnitAttr());

                // Pack the call arguments.
                for (int32_t i = 0; i < static_cast<int32_t>(callArgs.size()); ++i)
                {
                  auto arg = callArgs[i];
                  Value addr = rewriter.create<GetElementPointerOp>(
                    loc, ptrType, args, argPackType, ValueRange{}, SmallVector<int32_t>{ 0, i + 1 });
                  rewriter.create<StoreOp>(loc, arg, addr, UnitAttr(), UnitAttr());
                }

                // Create the func value.
                funcValue = rewriter.create<ZeroOp>(loc, funcType);
                funcValue = rewriter.create<InsertOp>(loc, funcType, funcPtr, 0, funcValue);
                funcValue = rewriter.create<InsertOp>(loc, funcType, args, 1, funcValue);
              }
              else
              {
                // The function can be used directly.
                Value funcPtr = rewriter.create<BitcastOp>(loc, ptrType, op.getCallee());
                funcValue = rewriter.create<ZeroOp>(loc, funcType);
                funcValue = rewriter.create<InsertOp>(loc, funcType, funcPtr, 0, funcValue);
              }
            })
          .Case(
            [&](GoStructType T)
            {
              if (!callArgs.empty())
              {
                // Collect the expected argument pack struct member types.
                // NOTE: The pointer to the callee is the first argument and the second is the
                // arguments
                //       pointer of the original func value.
                SmallVector<Type> argTypes = { ptrType, ptrType };
                llvm::append_range(argTypes, signature.getInputs());
                const auto argPackType = GoStructType::getBasic(context, argTypes);

                // Create a synthetic signature for the thunk.
                SmallVector<Type> inputs = { ptrType };
                llvm::append_range(inputs, signature.getInputs());
                const auto syntheticFnT =
                  FunctionType::get(context, inputs, signature.getResults());

                // Extract the function pointer from the callee func value.
                Value calleeFuncPtr = rewriter.create<ExtractOp>(loc, ptrType, 0, op.getCallee());

                // Extract the arguments pointer from the callee func value.
                Value argsPtr = rewriter.create<ExtractOp>(loc, ptrType, 1, op.getCallee());

                // Find or create a thunk to wrap the callee so that it's call arguments can be
                // unpacked properly.
                const auto funcSymbol = createGeneralCallThunk(rewriter, syntheticFnT, argPackType);

                // Get the thunk function by symbol.
                Value funcPtr = rewriter.create<AddressOfOp>(loc, ptrType, funcSymbol);

                // Allocate memory to store the call args.
                Value args =
                  rewriter.create<AllocaOp>(loc, ptrType, argPackType, 1, UnitAttr(), StringAttr());

                // Pack the pointer to the callee as the first call argument.
                rewriter.create<StoreOp>(loc, calleeFuncPtr, args, UnitAttr(), UnitAttr());

                // Pack the context pointer as the second call argument.
                Value addr = rewriter.create<GetElementPointerOp>(
                  loc, ptrType, args, argPackType, ValueRange{}, SmallVector<int32_t>{ 0, 1 });
                rewriter.create<StoreOp>(loc, argsPtr, addr, UnitAttr(), UnitAttr());

                // Pack the call arguments.
                for (int32_t i = 0; i < static_cast<int32_t>(callArgs.size()); ++i)
                {
                  auto arg = callArgs[i];
                  addr = rewriter.create<GetElementPointerOp>(
                    loc, ptrType, args, argPackType, ValueRange{}, SmallVector<int32_t>{ 0, i + 2 });
                  rewriter.create<StoreOp>(loc, arg, addr, UnitAttr(), UnitAttr());
                }

                // Create the new func value.
                funcValue = rewriter.create<ZeroOp>(loc, funcType);
                funcValue = rewriter.create<InsertOp>(loc, funcType, funcPtr, 0, funcValue);
                funcValue = rewriter.create<InsertOp>(loc, funcType, args, 1, funcValue);
              }
              else
              {
                // Use the callee func value directly.
                funcValue = op.getCallee();
              }
            })
          .Case(
            [&](InterfaceType T)
            {
              SmallVector<Type> argTypes = { T };
              llvm::append_range(argTypes, op.getCalleeOperands().getTypes());
              const auto argPackType = GoStructType::getBasic(context, argTypes);

              // Prepend the interface to the call args list.
              SmallVector<Value> callArgValues = { op.getCallee() };
              llvm::append_range(callArgValues, callArgs);

              // Find or create a thunk to wrap the callee so that it's call arguments can be
              // unpacked properly.
              const auto funcSymbol = createInterfaceCallThunk(
                rewriter, signature, T, *op.getMethodName(), callArgValues, argPackType);

              // Get the thunk function by symbol.
              Value funcPtr = rewriter.create<AddressOfOp>(loc, ptrType, funcSymbol);

              // Allocate memory to store the call args.
              Value args =
                rewriter.create<AllocaOp>(loc, ptrType, argPackType, 1, UnitAttr(), StringAttr());

              // Pack the call arguments.
              for (int32_t i = 0; i < static_cast<int32_t>(callArgValues.size()); ++i)
              {
                auto arg = callArgValues[i];
                Value addr = rewriter.create<GetElementPointerOp>(
                  loc, ptrType, args, argPackType, ValueRange{}, SmallVector<int32_t>{ 0, i });
                rewriter.create<StoreOp>(loc, arg, addr, UnitAttr(), UnitAttr());
              }

              // Create the new func value.
              funcValue = rewriter.create<ZeroOp>(loc, funcType);
              funcValue = rewriter.create<InsertOp>(loc, funcType, funcPtr, 0, funcValue);
              funcValue = rewriter.create<InsertOp>(loc, funcType, args, 1, funcValue);
            })
          .Default([&](Type) { assert(false && "unhandled callee type"); });

        // Replace the call operation with a runtime call to the scheduler.
        rewriter.replaceOpWithNewOp<RuntimeCallOp>(
          op, SmallVector<Type>{}, "runtime.addTask", SmallVector<Value>{ funcValue });
      });
  }
};

std::unique_ptr<mlir::Pass> createCallPass()
{
  return std::make_unique<CallPass>();
}
} // namespace mlir::go
