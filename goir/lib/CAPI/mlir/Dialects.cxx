#include "Go-c/mlir/Dialects.h"

#include <fstream>
#include <iostream>

#include <llvm/DebugInfo/DWARF/DWARFCompileUnit.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Support/raw_os_ostream.h>

#include <mlir/CAPI/Registration.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/Transforms/Passes.h>
#include <mlir/Dialect/Transform/IR/TransformDialect.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Import.h>
#include <mlir/Target/LLVMIR/ModuleTranslation.h>
#include <mlir/Transforms/Passes.h>

#include "Go/IR/GoAttrs.h"
#include "Go/IR/GoDialect.h"
#include "Go/IR/GoOps.h"
#include "Go/Transforms/Passes.h"
#include "Go/Transforms/TypeInfo.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Go, go, mlir::go::GoDialect)

struct IRPrinterConfig : public mlir::PassManager::IRPrinterConfig
{
  IRPrinterConfig(const llvm::StringRef dir)
    : mlir::PassManager::IRPrinterConfig(
        true,
        false,
        false,
        mlir::OpPrintingFlags().printGenericOpForm(false).enableDebugInfo())
    , m_dir(dir)
  {
    // Does nothing.
  }

  static auto New(const llvm::StringRef dir) { return std::make_unique<IRPrinterConfig>(dir); }

  void printBeforeIfEnabled(
    mlir::Pass* pass,
    mlir::Operation* operation,
    PrintCallbackFn printCallback) override
  {
    std::string fname = m_dir.str() + "/" + pass->getArgument().str() + ".before.mlir";
    std::ofstream os(fname);
    if (!os.is_open())
    {
      llvm::errs() << "Error opening file for writing: " << fname << "\n";
      return;
    }

    llvm::raw_os_ostream out(os);
    printCallback(out);
    os.close();
  }

  void printAfterIfEnabled(
    mlir::Pass* pass,
    mlir::Operation* operation,
    PrintCallbackFn printCallback) override
  {
    std::string fname = m_dir.str() + "/" + pass->getArgument().str() + ".after.mlir";
    std::ofstream os(fname);
    if (!os.is_open())
    {
      llvm::errs() << "Error opening file for writing: " << fname << "\n";
      return;
    }

    llvm::raw_os_ostream out(os);
    printCallback(out);
    os.close();
  }

private:
  const llvm::StringRef m_dir;
};

void mlirGoInitializeContext(MlirContext context)
{
}

MlirStringRef mlirModuleDump(MlirModule module)
{
  auto _module = unwrap(module);

  std::string result;
  llvm::raw_string_ostream out(result);

  _module.print(out, mlir::OpPrintingFlags().printGenericOpForm(false));

  // Allocate a buffer to copy the IR into
  char* buf = static_cast<char*>(malloc(result.size() + 1));
  strcpy(buf, result.data());

  // Return a string reference using the buffer
  return mlirStringRefCreate(buf, result.size());
}

bool mlirModuleDumpToFile(MlirModule module, MlirStringRef fname)
{
  auto _module = unwrap(module);
  auto _fname = unwrap(fname);

  std::error_code EC;
  ::llvm::raw_fd_ostream dest(_fname, EC, llvm::sys::fs::OF_Text);
  if (EC)
  {
    llvm::errs() << EC.message() << "\n";
    return false;
  }

  const auto flags = mlir::OpPrintingFlags().printGenericOpForm(false).enableDebugInfo();
  _module.print(dest, flags);
  dest.close();

  if (dest.has_error())
  {
    llvm::errs() << EC.message() << "\n";
    return false;
  }

  return true;
}

void mlirStringRefDestroy(MlirStringRef* ref)
{
  // Attempt to free the buffer referenced by the string reference
  free((void*)ref->data);
}

int mlirTypeHash(MlirType type)
{
  auto _type = unwrap(type);
  return mlir::hash_value(_type);
}

MlirAttribute mlirGoCreateTypeMetadata(MlirType type, MlirAttribute dict)
{
  const auto _type = mlir::TypeAttr::get(unwrap(type));
  const auto _dict = mlir::cast<mlir::DictionaryAttr>(unwrap(dict));
  return wrap(mlir::go::TypeMetadataAttr::get(_type.getContext(), _type, _dict));
}

MlirStringRef mlirGoGetTypeInfoSymbol(MlirType type, MlirStringRef prefix)
{
  return wrap(mlir::go::typeInfoSymbol(unwrap(type), unwrap(prefix).str()));
}

MlirOperation mlirCreateUnrealizedConversionCastOp(
  MlirContext context,
  MlirType type,
  MlirValue value,
  MlirLocation location)
{
  auto _context = unwrap(context);
  auto _type = unwrap(type);
  auto _value = unwrap(value);
  auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = builder.create<mlir::UnrealizedConversionCastOp>(_location, _type, _value);
  return wrap(op);
}

void mlirGoBindRuntimeType(MlirModule module, MlirStringRef mnemonic, MlirType runtimeType)
{
  const auto _module = unwrap(module);
  const auto _mnemonic = unwrap(mnemonic);
  const auto _runtimeType = unwrap(runtimeType);

  llvm::SmallVector<mlir::NamedAttribute> entries;
  if (_module->hasAttr("go.runtimeTypes"))
  {
    const auto map = mlir::dyn_cast<mlir::DictionaryAttr>(_module->getAttr("go.runtimeTypes"));
    for (auto entry : map)
    {
      entries.emplace_back(entry.getName(), entry.getValue());
    }
  }

  entries.emplace_back(
    mlir::StringAttr::get(_module->getContext(), _mnemonic), mlir::TypeAttr::get(_runtimeType));
  auto result = mlir::DictionaryAttr::get(_module->getContext(), entries);
  _module->setAttr("go.runtimeTypes", result);
}

void mlirGoBindRuntimeTypeToType(MlirModule module, MlirType primitiveType, MlirType runtimeType)
{
  const auto _module = unwrap(module);
  const auto _primitiveType = unwrap(primitiveType);
  const auto _runtimeType = unwrap(runtimeType);
  const auto _ctx = _module->getContext();

  mlir::DenseSet<mlir::Type> types;

  std::vector<mlir::DataLayoutEntryInterface> dltiEntries;
  if (
    auto originalSpec =
      _module->getAttrOfType<mlir::DataLayoutSpecAttr>(mlir::DLTIDialect::kDataLayoutAttrName))
  {
    dltiEntries = originalSpec.getEntries().vec();
    for (const auto& entry : dltiEntries)
    {
      if (auto type = llvm::dyn_cast_if_present<mlir::Type>(entry.getKey()))
      {
        types.insert(type);
      }
    }
  }

  // Prevent duplicates. (Although this should be unreachable).
  if (!types.insert(_primitiveType).second)
  {
    // There is already an entry for this type.
    return;
  }

  // Create a spec entry for this runtime type that is bound to the primitive type.
  const auto primitiveTypeAttr = mlir::TypeAttr::get(_primitiveType);
  const auto runtimeTypeAttr = mlir::TypeAttr::get(_runtimeType);
  dltiEntries.emplace_back(
    mlir::go::RuntimeTypeEntryAttr::get(_ctx, primitiveTypeAttr, runtimeTypeAttr));

  // Update the data layout spec.
  const auto spec = mlir::DataLayoutSpecAttr::get(_ctx, dltiEntries);
  _module->setAttr(mlir::DLTIDialect::kDataLayoutAttrName, spec);
}

void mlirGoSetTargetDataLayout(MlirModule module, LLVMTargetDataRef layout)
{
  auto _module = unwrap(module);
  auto _layout = llvm::unwrap(layout);
  _module->setAttr(
    "llvm.data_layout",
    ::mlir::StringAttr::get(_module->getContext(), _layout->getStringRepresentation()));

  auto ctx = _module->getContext();

  // Also import this data layout and apply it to the module
  auto spec = mlir::translateDataLayout(*_layout, _module->getContext());
  auto entries = spec.getEntries().vec();

  // Set the index type to be the same width as a pointer in the target machine.
  entries.emplace_back(
    mlir::DataLayoutEntryAttr::get(
      mlir::IndexType::get(ctx),
      mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), _layout->getPointerSizeInBits())));

  spec = mlir::DataLayoutSpecAttr::get(ctx, entries);

  _module->setAttr(mlir::DLTIDialect::kDataLayoutAttrName, spec);
}

void mlirGoSetTargetTriple(MlirModule module, MlirStringRef triple)
{
  auto _module = unwrap(module);
  auto _triple = unwrap(triple);
  _module->setAttr("llvm.target_triple", ::mlir::StringAttr::get(_module->getContext(), _triple));
}

MlirLogicalResult mlirCanonicalizeModule(MlirModule module)
{
  auto _module = unwrap(module);
  auto pm = mlir::PassManager::on<mlir::ModuleOp>(_module->getContext());
  pm.addPass(mlir::createCanonicalizerPass());
  return wrap(pm.run(_module));
}

MlirLogicalResult
mlirGoOptimizeModule(MlirModule module, MlirStringRef name, MlirStringRef outputDir, bool debug)
{
  auto _module = unwrap(module);
  llvm::Twine _name = unwrap(name);
  auto _outputDir = unwrap(outputDir);
  auto pm = mlir::PassManager::on<mlir::ModuleOp>(_module->getContext());
  pm.enableVerifier(true);

  if (debug)
  {
    pm.getContext()->disableMultithreading();
    pm.enableIRPrinting(IRPrinterConfig::New(_outputDir));
    pm.enableStatistics();
    pm.enableTiming();
  }

  // ─────────────────────────────────────────────
  // Phase 1: Top-level module passes
  // ─────────────────────────────────────────────
  pm.addPass(mlir::go::createPreprocessingPass());
  pm.addPass(mlir::go::createCallPass());
  pm.addPass(mlir::go::createAttachDebugInfoPass());
  pm.addPass(mlir::go::createGlobalConstantsPass());
  pm.addPass(mlir::go::createGlobalInitializerPass());

  // ─────────────────────────────────────────────
  // Phase 2: Per-function Go semantic passes
  // ─────────────────────────────────────────────
  {
    auto& nestedFuncPM = pm.nest<mlir::go::FuncOp>();
    nestedFuncPM.addPass(mlir::go::createHeapEscapePass());
    nestedFuncPM.addPass(mlir::go::createFunctionPass());
  }

  // ─────────────────────────────────────────────
  // Phase 3: Lower Go-specific ops (all ops, not just functions)
  // ─────────────────────────────────────────────
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::go::createLowerToCorePass());

  // ─────────────────────────────────────────────
  // Phase 4: LLVM lowering on the full module
  // ─────────────────────────────────────────────
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::go::createLowerToLLVMPass());

  // ─────────────────────────────────────────────
  // Phase 5: LLVM export + cleanup
  // ─────────────────────────────────────────────
  pm.addNestedPass<mlir::LLVM::LLVMFuncOp>(mlir::LLVM::createLegalizeForExportPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  return wrap(pm.run(_module));
}

MlirAttribute mlirGetLLVMLinkageAttr(MlirContext context, MlirStringRef linkage)
{
  auto _context = unwrap(context);
  auto _linkage = unwrap(linkage);
  if (auto value = mlir::LLVM::linkage::symbolizeLinkage(_linkage); value)
  {
    return wrap(mlir::LLVM::LinkageAttr::get(_context, *value));
  }
  assert(false && "unreachable");
}

void mlirInitModuleTranslation(MlirContext context)
{
  auto _context = unwrap(context);

  // Register the translation to LLVM IR with the MLIR context.
  mlir::registerBuiltinDialectTranslation(*_context);
  mlir::registerLLVMDialectTranslation(*_context);
}

LLVMModuleRef
mlirTranslateModuleToLLVMIR(MlirModule module, LLVMContextRef llvmContext, MlirStringRef name)
{
  auto _module = unwrap(module);
  auto _llvmContext = llvm::unwrap(llvmContext);
  auto _name = unwrap(name);
  return llvm::wrap(translateModuleToLLVMIR(_module, *_llvmContext, _name).release());
}

MlirAttribute mlirGoCreateTypeMetadataEntryAttr(MlirType type, MlirAttribute dict)
{
  const auto _type = mlir::TypeAttr::get(unwrap(type));
  const auto _dict = mlir::cast<mlir::DictionaryAttr>(unwrap(dict));
  return wrap(mlir::go::TypeMetadataEntryAttr::get(_type.getContext(), _type, _dict));
}

MlirAttribute mlirGoCreateTypeMetadataDictionaryAttr(
  MlirContext context,
  int nEntries,
  MlirAttribute* entries)
{
  const auto _context = unwrap(context);
  mlir::SmallVector<mlir::Attribute> values;
  (void)unwrapList(nEntries, entries, values);

  mlir::DenseSet<mlir::go::TypeMetadataEntryAttr> _entries;
  for (auto& value : values)
  {
    _entries.insert(mlir::cast<mlir::go::TypeMetadataEntryAttr>(value));
  }

  const auto _arrayAttr = mlir::go::TypeMetadataEntryArrayAttr::get(
    _context, mlir::SmallVector<mlir::go::TypeMetadataEntryAttr>(_entries.begin(), _entries.end()));
  return wrap(mlir::go::TypeMetadataDictionaryAttr::get(_context, _arrayAttr));
}

MlirBlock mlirRegionGetLastBlock(MlirRegion region)
{
  auto _region = unwrap(region);
  if (!_region->empty())
    return wrap(&_region->getBlocks().back());
  return wrap(static_cast<mlir::Block*>(nullptr));
}

MlirLogicalResult mlirVerifyModule(MlirModule module)
{
  auto _module = unwrap(module);
  return wrap(mlir::verify(_module, true));
}

MlirAttribute
mlirGoCreateComplexNumberAttr(MlirContext context, MlirType type, double real, double imag)
{
  auto _context = unwrap(context);
  auto _type = unwrap(type);
  auto floatType = mlir::cast<mlir::ComplexType>(_type).getElementType();
  return wrap(
    mlir::go::ComplexNumberAttr::get(
      _context, mlir::FloatAttr::get(floatType, real), mlir::FloatAttr::get(floatType, imag)));
}

bool mlirOperationHasNoMemoryEffect(MlirOperation op)
{
  auto _op = unwrap(op);
  auto face = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(_op);
  if (!face || !face.hasNoEffect())
  {
    return false;
  }
  return true;
}

MlirOperation mlirValueGetDefiningOperation(MlirValue value)
{
  auto _value = unwrap(value);
  return wrap(_value.getDefiningOp());
}

MlirBlock
mlirBlockCreate2(int nArgs, MlirType* args, int nLocations, MlirLocation* locations)
{
  assert(nArgs == nLocations);
  return mlirBlockCreate(nArgs, args, locations);
}

MlirAttribute mlirDistinctAttrGet(MlirAttribute attr)
{
  auto _attr = unwrap(attr);
  return wrap(mlir::DistinctAttr::create(_attr));
}

void mlirGoBlockDumpTail(MlirBlock block, int count)
{
  auto _block = unwrap(block);
  auto it = _block->rbegin();
  while (count > 0)
  {
    it->dump();
    --count;
    ++it;
    if (it == _block->rend())
    {
      break;
    }
  }
}

MlirAttribute mlirGoCreateAsmConstraintAttr(
  MlirContext context,
  MlirStringRef registerClass,
  MlirGoAsmConstraintDirection direction,
  MlirStringRef alias,
  int operandIndex,
  bool reserve)
{
  mlir::go::AsmConstraintDirectionAttr _dir;
  mlir::StringAttr _alias;
  mlir::IntegerAttr _operandIndex;
  mlir::UnitAttr _reserve;

  const auto _context = unwrap(context);
  const auto _registerClass = mlir::StringAttr::get(_context, unwrap(registerClass));
  switch (direction)
  {
    case MlirGoAsmConstraintDirection::In:
      _dir = mlir::go::AsmConstraintDirectionAttr::get(
        _context, mlir::UnitAttr::get(_context), mlir::UnitAttr());
      break;
    case MlirGoAsmConstraintDirection::Out:
      _dir = mlir::go::AsmConstraintDirectionAttr::get(
        _context, mlir::UnitAttr(), mlir::UnitAttr::get(_context));
      break;
    case MlirGoAsmConstraintDirection::InOut:
      _dir = mlir::go::AsmConstraintDirectionAttr::get(
        _context, mlir::UnitAttr::get(_context), mlir::UnitAttr::get(_context));
      break;
  }

  if (alias.length > 0)
  {
    _alias = mlir::StringAttr::get(_context, unwrap(alias));
  }

  if (operandIndex >= 0)
  {
    _operandIndex = mlir::IntegerAttr::get(mlir::IntegerType::get(_context, 32), operandIndex);
  }

  if (reserve)
  {
    _reserve = mlir::UnitAttr::get(_context);
  }

  const auto attr = mlir::go::AsmConstraintAttr::get(
    _context, _registerClass, _dir, _alias, _operandIndex, _reserve);
  return wrap(attr);
}