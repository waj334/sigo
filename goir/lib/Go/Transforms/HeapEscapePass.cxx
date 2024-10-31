#include <llvm/ADT/TypeSwitch.h>

#include "Go/IR/GoOps.h"
#include "Go/Transforms/Passes.h"

namespace mlir::go
{
struct HeapEscapePass : public PassWrapper<HeapEscapePass, OperationPass<mlir::go::FuncOp>>
{
  enum class Result
  {
    DoesNotEscape = 0,
    EscapesToHeap = 1
  };

  mlir::DenseMap<Operation*, Result> results;
  mlir::DenseSet<Operation*> visited;

  void runOnOperation() override
  {
    auto funcOp = getOperation();

    // Lower "new" built-in calls to alloca operations.
    OpBuilder builder(&this->getContext());
    SmallVector<Operation*> opsToRemove;
    opsToRemove.reserve(64);

    funcOp->walk(
      [&](BuiltInCallOp op)
      {
        if (op.getCallee() == "new")
        {
          const auto loc = op.getLoc();
          const auto elementType =
            *go::cast<PointerType>(op.getResult(0).getType()).getElementType();

          // Create the replacement alloca operation.
          builder.setInsertionPoint(op);
          const auto ptrType = PointerType::get(funcOp.getContext(), elementType);
          auto allocaOp = builder.create<AllocaOp>(
            loc, ptrType, elementType, 1, mlir::UnitAttr(), mlir::StringAttr());
          op.replaceAllUsesWith(allocaOp);

          // Remove the built-in call later.
          opsToRemove.push_back(op);
        }
      });

    // Now actually remove any operations.
    for (auto op : opsToRemove)
    {
      op->erase();
    }

    // Analyze all top level allocations.
    funcOp.walk(
      [&](AllocaOp allocaOp)
      {
        if (allocaOp->hasAttr("isArgument"))
        {
          // Skip function arguments.
          return;
        }

        DenseSet<Operation*> visited;
        if (analyzeOperation(allocaOp, visited) == Result::EscapesToHeap)
        {
          // This stack allocation has been determined to escape from the heap.
          allocaOp.setHeap(true);
        }
      });
  }

  Result analyzeOperation(Operation* op, DenseSet<Operation*>& visited, bool considerLoads = false)
  {
    if (visited.contains(op))
    {
      return Result::DoesNotEscape;
    }

    visited.insert(op);

    // Analyze this operation's users.
    for (auto user : op->getUsers())
    {
      const Result result =
        mlir::TypeSwitch<Operation*, Result>(user)
          .Case(
            [&](AllocaOp allocaOp)
            {
              if (allocaOp->hasAttr("isArgument"))
              {
                // This alloca represents a function parameter, handle accordingly.
                return Result::EscapesToHeap;
              }
              return Result::DoesNotEscape;
            })
          .Case([&](AddressOfOp) { return Result::EscapesToHeap; })
          .Case([&](CallOp) { return Result::EscapesToHeap; })
          .Case([&](GetElementPointerOp gepOp)
                { return this->analyzeOperation(gepOp.getValue().getDefiningOp(), visited); })
          .Case(
            [&](InsertOp insertOp)
            {
              return this->analyzeOperation(insertOp, visited);
            })
          .Case([&](LoadOp loadOp)
          {
            if (considerLoads)
            {
              return this->analyzeOperation(loadOp, visited);
            }
            return Result::DoesNotEscape;
          })
          .Case([&](ReturnOp) { return Result::EscapesToHeap; })
          .Case([&](SliceAddrOp sliceAddrOp)
                { return this->analyzeOperation(sliceAddrOp.getSlice().getDefiningOp(), visited); })
          .Case(
            [&](StoreOp storeOp)
            {
              // Evaluate whether the address being stored to will cause the value to escape the
              // current function.
              auto [escapes, baseAddrOp] = this->isEscapingAddress(storeOp.getAddr());
              if (escapes)
              {
                // The address is known to cause an escape.
                return Result::EscapesToHeap;
              }

              if (baseAddrOp)
              {
                // Evaluate if the value of the address being stored to escapes through another
                // operation. Load operations should be examined to determine if the base address
                // is derived from a value that already escapes.
                return this->analyzeOperation(baseAddrOp, visited, true);
              }

              return Result::DoesNotEscape;
            })
          .Default([&](auto) { return Result::DoesNotEscape; });
      if (result == Result::EscapesToHeap)
      {
        return result;
      }
    }
    return Result::DoesNotEscape;
  }

  std::pair<bool, Operation*> isEscapingAddress(const mlir::Value addr) const
  {
    return mlir::TypeSwitch<Operation*, std::pair<bool, Operation*>>(addr.getDefiningOp())
      .Case(
        [&](GetElementPointerOp gepOp) -> std::pair<bool, Operation*>
        {
          // Analyze the base address.
          return this->isEscapingAddress(gepOp.getValue());
        })
      .Case(
        [&](AddressOfOp addressOp) -> std::pair<bool, Operation*>
        {
          // This is a global.
          return { true, nullptr };
        })
      .Case(
        [&](SliceAddrOp sliceAddrOp) -> std::pair<bool, Operation*>
        { return this->isEscapingAddress(sliceAddrOp.getSlice()); })
      .Case(
        [&](LoadOp loadOp) -> std::pair<bool, Operation*>
        { return this->isEscapingAddress(loadOp.getOperand()); })
      .Case(
        [&](AllocaOp allocaOp) -> std::pair<bool, Operation*>
        {
          if (allocaOp->hasAttr("isArgument"))
          {
            // The address is derived from a function argument.
            return { true, nullptr };
          }
          return { false, allocaOp };
        })
      .Default([&](auto op) -> std::pair<bool, Operation*> { return { false, nullptr }; });
  }

  StringRef getArgument() const final { return "go-heap-escape-pass"; }

  StringRef getDescription() const final
  {
    return "Analyzes stack allocations that escape the stack.";
  }

  void getDependentDialects(DialectRegistry& registry) const override
  {
    registry.insert<GoDialect>();
    registry.insert<func::FuncDialect>();
  }
};

std::unique_ptr<mlir::Pass> createHeapEscapePass()
{
  return std::make_unique<HeapEscapePass>();
}
} // namespace mlir::go
