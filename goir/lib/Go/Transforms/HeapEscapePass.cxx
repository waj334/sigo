#include <llvm/ADT/TypeSwitch.h>

#include "Go/IR/GoOps.h"
#include "Go/Transforms/Passes.h"

namespace mlir::go
{
struct HeapEscapePass : public PassWrapper<HeapEscapePass, OperationPass<func::FuncOp>>
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

          // Remove the built-in call.
          op.erase();
        }
      });

    // Analyze all top level allocations.
    funcOp.walk(
      [&](AllocaOp allocaOp)
      {
        if (analyzeOperation(allocaOp) == Result::EscapesToHeap)
        {
          // This stack allocation has been determined to escape from the heap.
          allocaOp.setHeap(true);
        }
      });
  }

  Result analyzeOperation(Operation* op, bool considerLoads = false, bool ignoreCache = false)
  {
    if (!ignoreCache)
    {
      // Look up in the cache first.
      if (const auto it = this->results.find(op); it != this->results.end())
      {
        // Return the cached result.
        return it->second;
      }
    }

    // Update visited set.
    this->visited.insert(op);

    const Result result =
      llvm::TypeSwitch<Operation*, Result>(op)
        .Case(
          [&](AllocaOp nestedAllocaOp)
          {
            for (auto nestedAllocaUser : nestedAllocaOp->getUsers())
            {
              // Skip operations that were already visited.
              if (this->visited.find(nestedAllocaUser) != this->visited.end())
              {
                continue;
              }

              // Analyze the user.
              if (analyzeOperation(nestedAllocaUser) == Result::EscapesToHeap)
              {
                // Stop analyzing uses.
                return Result::EscapesToHeap;
              }
            }
            return Result::DoesNotEscape;
          })
        .Case(
          [&](func::CallOp)
          {
            // Pointers passed to functions should be assumed to escape.
            return Result::EscapesToHeap;
          })
        .Case(
          [&](func::CallIndirectOp)
          {
            // The actual callee is unknown, so this allocation should be moved to the heap
            // conservatively.
            return Result::EscapesToHeap;
          })
        .Case(
          [&](InsertOp insertOp)
          {
            /*auto definingOp = insertOp.getAggregate().getDefiningOp();
            if (auto loadOp = mlir::dyn_cast<LoadOp>(definingOp)) {
                // Analyze the address being loaded from.
                if (auto loadedAllocation =
            mlir::dyn_cast<AllocaOp>(loadOp.getOperand().getDefiningOp())) { return
            this->analyzeOperation(loadedAllocation);
                }
            }

            // Check if the defining operation of the aggregate does not escape.
            if (analyzeOperation(definingOp) == Result::DoesNotEscape) {
                // Check if any user of the insert operation value should cause an escape.
                for (auto user: insertOp->getUsers()) {
                    // Skip operations that were already visited.
                    if (this->visited.find(user) != this->visited.end()) {
                        continue;
                    }

                    if (this->analyzeOperation(user) == Result::EscapesToHeap) {
                        return Result::EscapesToHeap;
                    }
                }
            }
            */

            /*
        // Follow the chain of subsequent inserts taking load operations into account.
        for (auto user: insertOp->getUsers()) {
            // Skip operations that were already visited.
            //if (this->visited.find(user) != this->visited.end()) {
            //    continue;
            //}

            if (this->analyzeOperation(user, true, true) == Result::EscapesToHeap) {
                return Result::EscapesToHeap;
            }
        }
        return Result::DoesNotEscape;
        */

            // Assume stack pointers inserted into some struct should escape to the heap.
            // TODO: Need to check if the struct value is actually returned from the current
            // function.
            return Result::EscapesToHeap;
          })
        .Case(
          [&](InterfaceCallOp)
          {
            // The actual callee is unknown, so this allocation should be moved to the heap
            // conservatively.
            return Result::EscapesToHeap;
          })
        .Case(
          [&](LoadOp loadOp)
          {
            if (considerLoads)
            {
              // Check if the loaded value escapes.
              for (auto user : loadOp->getUsers())
              {
                // Skip operations that were already visited.
                if (this->visited.find(user) != this->visited.end())
                {
                  continue;
                }

                if (this->analyzeOperation(user) == Result::EscapesToHeap)
                {
                  return Result::EscapesToHeap;
                }
              }
            }
            return Result::DoesNotEscape;
          })
        .Case(
          [&](func::ReturnOp)
          {
            // The address is returned from the current function.
            return Result::EscapesToHeap;
          })
        .Case(
          [&](StoreOp storeOp)
          {
            return llvm::TypeSwitch<Operation*, Result>(storeOp.getAddr().getDefiningOp())
              .Case(
                [&](AddressOfOp)
                {
                  // The address of the allocation is being stored in a global variable, so it
                  // escapes to the heap.
                  return Result::EscapesToHeap;
                })
              .Case(
                [&](AllocaOp nestedAllocaOp)
                {
                  // The allocation escapes if the address it is being stored to escapes.
                  return analyzeOperation(nestedAllocaOp);
                })
              .Default([&](Operation*) { return Result::DoesNotEscape; });
          })
        .Default([&](Operation*) { return Result::DoesNotEscape; });

    // Cache the result and then return.
    this->results[op] = result;
    return result;
  }

  /*
  using Use = std::pair<Operation *, AllocaOp>;
  using WorkQueue = std::queue<Use>;
  using VisitedSet = std::unordered_set<Operation *>;
  using ParameterResultMap = std::unordered_map<unsigned, Result>;
  using FunctionParameterMap = std::unordered_map<FuncOp, ParameterResultMap>;

  WorkQueue queue;
  VisitedSet visited;

  void runOnOperation() override {
      auto funcOp = getOperation();
      auto module = funcOp->getParentOfType<ModuleOp>();


      // Process the work queue.
      while (!queue.empty()) {
          // Pop the use off of the queue.
          auto [user, allocaOp] = queue.front();
          queue.pop();

          if (auto [_, ok] = visited.insert(user); !ok) {
              // This operation has already been analyzed.
              continue;
          }

          // Analyze the user.
          if (analyzeUser(module, allocaOp, user) == Result::EscapesToHeap) {
              // This stack allocation has been determined to escape from the heap.
              allocaOp.setHeap(true);
          }
      }
  }

  Result analyzeFunction(ModuleOp module, func::FuncOp funcOp) {
      // First, analyze function parameters that are pointers.
      for (size_t arg = 0; arg < funcOp.getNumArguments(); arg++) {
          auto argValue = funcOp.getArgument(arg);
          const auto argType = argValue.getType();
          if (go::isa<PointerType>(argType)) {
              // The only use of the argument value should be an alloca.
              for (auto user: argValue.getUsers()) {
                  if (auto allocaOp = mlir::dyn_cast<AllocaOp>(user)) {
                      for (auto allocaUser: allocaOp->getUsers()) {
                          queue.emplace(allocaUser, allocaOp);
                      }
                  }
              }
          }
      }

      // Push all users of an alloca's value to the work queue.
      funcOp.walk([&](AllocaOp allocaOp) {
          for (auto user: allocaOp->getUsers()) {
              queue.emplace(user, allocaOp);
          }
      });
  }

  Result analyzeUser(ModuleOp module, AllocaOp allocaOp, Operation *user) {
      return llvm::TypeSwitch<Operation *, Result>(user)
              .Case([&](func::CallOp callOp) {
                  const auto sym = callOp.getCallee();
                  auto fnOp = module.lookupSymbol<func::FuncOp>(sym);
                  const auto callArgs = callOp.getArgOperands();

                  // NOTE: This analysis only is applicable for functions that are called in exactly
  one location
                  //       in the program. Otherwise, it

                  // Determine which argument is the pointer.
                  for (size_t i = 0; i < callArgs.size(); i++) {
                      if (callArgs[i].getDefiningOp() == allocaOp) {
                          // Analyze the usage of this argument.
                          if (this->analyzeCalleeArg(module, allocaOp, fnOp, i) ==
  Result::EscapesToHeap) { return Result::EscapesToHeap;
                          }
                      }
                  }
                  return Result::DoesNotEscape;
              })
              .Case([&](func::CallIndirectOp callIndirectOp) {
                  // The actual callee is unknown, so this allocation should be moved to the heap
                  // conservatively.
                  return Result::EscapesToHeap;
              })
              .Case([&](InterfaceCallOp) {
                  // The actual callee is unknown, so this allocation should be moved to the heap
                  // conservatively.
                  return Result::EscapesToHeap;
              })
              .Case([&](func::ReturnOp) {
                  // The address is returned from the current function.
                  return Result::EscapesToHeap;
              })
              .Case([&](StoreOp storeOp) {
                  return this->analyzeStoreOp(module, allocaOp, storeOp);
              }).Default([&](Operation *) {
                  return Result::DoesNotEscape;
              });
  }

  Result analyzeStoreOp(ModuleOp module, AllocaOp allocaOp, StoreOp storeOp) {
      // The value being stored must be a pointer type.
      if (!go::isa<PointerType>(storeOp.getValue().getType())) {
          return Result::DoesNotEscape;
      }

      // Analyze where the value is being stored to. This function should cause the location being
  stored to be
      // analyzed to determine if it escapes to the heap (I.E. storing to globals).
      return llvm::TypeSwitch<Operation *, Result>(storeOp.getAddr().getDefiningOp())
              .Case([&](AddressOfOp) {
                  // The storage location is a global value.
                   // var myGlobal *int
                   // func fn() {
                   //    var value int
                   //    myGlobal = &value
                   //
                  return Result::EscapesToHeap;
              })
              .Case([&](AllocaOp op) {
                  // The allocation may escape to the heap indirectly. Analyze the users of this
  alloca op.
                  // func fn() {
                  //     var x int
                  //     var p *int = &x
                  //     var pp **int = &p  // Indirect heap escape
                  //     return pp  // This would be handled in analyzeReturnOp
                  // }
                  for (auto user: op->getUsers()) {
                      if (user == storeOp || visited.find(user) != visited.end()) {
                          // Skip current store operation.
                          continue;
                      }
                      queue.emplace(user, allocaOp);
                  }
                  return Result::DoesNotEscape;
              })
              .Default([&](Operation *) {
                  return Result::DoesNotEscape;
              });
  }

  Result analyzeCalleeArg(ModuleOp module, AllocaOp allocaOp, func::FuncOp fnOp, uintptr_t arg) {
      // First the block argument value representing the function parameter.
      auto argValue = fnOp.getArgument(arg);

      // Analyze the uses of the argument value.
      for (auto user: argValue.getUsers()) {
          if (this->analyzeUser(module, allocaOp, user) == Result::EscapesToHeap) {
              return Result::EscapesToHeap;
          }
      }
      return Result::DoesNotEscape;
  }
  */

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
