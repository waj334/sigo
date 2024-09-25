
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include "Go/IR/GoOps.h"

namespace mlir::go
{

struct ExtractTypeMetadataPass : PassWrapper<ExtractTypeMetadataPass, OperationPass<mlir::ModuleOp>>
{
  using TypeMetadataMap = mlir::DenseMap<Type, mlir::DictionaryAttr>;
  void runOnOperation() final
  {
    auto module = getOperation();
    auto context = module.getContext();
    TypeMetadataMap metadataMap;

    // Generate type metadata for declared types.
    module.walk([&](DeclareTypeOp op)
    {
      this->emitTypeMetadata(op.getDeclaredType(), metadataMap);
    });

    // Emit a dictionary of type information into the module.
    SmallVector<mlir::NamedAttribute> namedAttrs;
    namedAttrs.reserve(metadataMap.size());
    for (const auto [type, metadata] : metadataMap)
    {

    }

    auto typeMetadataDict = mlir::DictionaryAttr::get(context);
    module->setAttr("typeMetadata", typeMetadataDict);
  }

  void emitTypeMetadata(const mlir::Type& type, TypeMetadataMap& entries)
  {
    auto metadata = mlir::DictionaryAttr::get(type.getContext());

    // Insert an entry for this type into the entries map.
    entries.insert({type, metadata});
  }
};

} // namespace mlir::go