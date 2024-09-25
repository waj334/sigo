#include <fstream>

#include <llvm/Support/raw_os_ostream.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

using namespace std::filesystem;

namespace mlir::go
{
struct DumpToFilePass : PassWrapper<DumpToFilePass, OperationPass<ModuleOp>>
{
  std::string name;
  std::string dir;

  StringRef getArgument() const final { return "dump-to-file"; }

  StringRef getDescription() const final
  {
    return "Dump To File Pass - Dumps the module to a file";
  }

  void runOnOperation() final
  {
    auto module = this->getOperation();

    const std::string fname = dir + "/" + name + ".mlir";
    std::ofstream os(fname);
    if (!os.is_open())
    {
      llvm::errs() << "Error opening file for writing: " << fname << "\n";
      return;
    }

    llvm::raw_os_ostream out(os);

    const auto flags = mlir::OpPrintingFlags().printGenericOpForm(false).enableDebugInfo();
    module.print(out, flags);
    os.close();
  }
};

std::unique_ptr<mlir::Pass> createDumpToFilePass(StringRef name, StringRef dir)
{
  auto result = std::make_unique<DumpToFilePass>();
  result->name = name;
  result->dir = dir;
  return result;
}
} // namespace mlir::go
