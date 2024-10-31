#include "Go-c/mlir/Dialects.h"

#include "Go/IR/GoAttrs.h"
#include "Go/IR/GoDialect.h"
#include "Go/IR/GoOps.h"
#include "Go/Transforms/TypeInfo.h"
#include "Go/Transforms/Passes.h"

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/ToolOutputFile.h>
#include <mlir/CAPI/Registration.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/DebugInfo/DWARF/DWARFCompileUnit.h>
#include <llvm/IR/DataLayout.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/LLVMIR/Transforms/Passes.h>
#include <mlir/Dialect/Transform/IR/TransformDialect.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Import.h>
#include <mlir/Target/LLVMIR/ModuleTranslation.h>

#include <fstream>
#include <iostream>
#include <llvm/Support/raw_os_ostream.h>

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Go, go, mlir::go::GoDialect)

struct IRPrinterConfig : public mlir::PassManager::IRPrinterConfig {
    IRPrinterConfig(const llvm::StringRef dir)
        : mlir::PassManager::IRPrinterConfig(true, false, false,
                                             mlir::OpPrintingFlags().printGenericOpForm(false).enableDebugInfo()),
          m_dir(dir) {
        // Does nothing.
    }

    static auto New(const llvm::StringRef dir) { return std::make_unique<IRPrinterConfig>(dir); }

    void printBeforeIfEnabled(mlir::Pass *pass, mlir::Operation *operation, PrintCallbackFn printCallback) override {
        std::string fname = m_dir.str() + "/" + pass->getArgument().str() + ".before.mlir";
        std::ofstream os(fname);
        if (!os.is_open()) {
            llvm::errs() << "Error opening file for writing: " << fname << "\n";
            return;
        }

        llvm::raw_os_ostream out(os);
        printCallback(out);
        os.close();
    }

    void printAfterIfEnabled(mlir::Pass *pass, mlir::Operation *operation, PrintCallbackFn printCallback) override {
        std::string fname = m_dir.str() + "/" + pass->getArgument().str() + ".after.mlir";
        std::ofstream os(fname);
        if (!os.is_open()) {
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
  mlir::MLIRContext* _context = unwrap(context);

  // TODO: Initialize any module-level interfaces here.
}

MlirStringRef mlirModuleDump(MlirModule module) {
    auto _module = unwrap(module);

    std::string result;
    llvm::raw_string_ostream out(result);

    _module.print(out, mlir::OpPrintingFlags().printGenericOpForm(false));

    // Allocate a buffer to copy the IR into
    char *buf = static_cast<char *>(malloc(result.size() + 1));
    strcpy(buf, result.data());

    // Return a string reference using the buffer
    return mlirStringRefCreate(buf, result.size());
}

bool mlirModuleDumpToFile(MlirModule module, MlirStringRef fname) {
    auto _module = unwrap(module);
    auto _fname = unwrap(fname);

    std::error_code EC;
    ::llvm::raw_fd_ostream dest(_fname, EC, llvm::sys::fs::OF_TextWithCRLF);
    if (EC) {
        //*ErrorMessage = strdup(EC.message().c_str());
        return false;
    }

    const auto flags = mlir::OpPrintingFlags().printGenericOpForm(false).enableDebugInfo();
    _module.print(dest, flags);
    dest.close();

    if (dest.has_error()) {
        // std::string E = "Error printing to file: " + dest.error().message();
        //*ErrorMessage = strdup(E.c_str());
        return false;
    }

    return true;
}

void mlirStringRefDestroy(MlirStringRef *ref) {
    // Attempt to free the buffer referenced by the string reference
    free((void *) ref->data);
}

intptr_t mlirTypeHash(MlirType type) {
    auto _type = unwrap(type);
    return mlir::hash_value(_type);
}

MlirAttribute mlirGoCreateTypeMetadata(MlirType type, MlirAttribute dict) {
    const auto _type = mlir::TypeAttr::get(unwrap(type));
    const auto _dict = mlir::cast<mlir::DictionaryAttr>(unwrap(dict));
    return wrap(mlir::go::TypeMetadataAttr::get(_type.getContext(), _type, _dict));
}

MlirStringRef mlirGoGetTypeInfoSymbol(MlirType type, MlirStringRef prefix) {
    return wrap(mlir::go::typeInfoSymbol(unwrap(type), unwrap(prefix).str()));
}

MlirOperation mlirCreateUnrealizedConversionCastOp(MlirContext context, MlirType type, MlirValue value,
                                                   MlirLocation location) {
    auto _context = unwrap(context);
    auto _type = unwrap(type);
    auto _value = unwrap(value);
    auto _location = unwrap(location);

    mlir::OpBuilder builder(_context);
    mlir::Operation *op = builder.create<mlir::UnrealizedConversionCastOp>(_location, _type, _value);
    return wrap(op);
}

void mlirGoBindRuntimeType(MlirModule module, MlirStringRef mnemonic, MlirType runtimeType) {
    auto _module = unwrap(module);
    auto _mnemonic = unwrap(mnemonic);
    auto _runtimeType = unwrap(runtimeType);

    llvm::SmallVector<mlir::NamedAttribute> entries;

    if (_module->hasAttr("go.runtimeTypes")) {
        auto map = mlir::dyn_cast<mlir::DictionaryAttr>(_module->getAttr("go.runtimeTypes"));
        for (auto entry: map) {
            entries.emplace_back(entry.getName(), entry.getValue());
        }
    }

    entries.emplace_back(mlir::StringAttr::get(_module->getContext(), _mnemonic), mlir::TypeAttr::get(_runtimeType));
    auto result = mlir::DictionaryAttr::get(_module->getContext(), entries);
    _module->setAttr("go.runtimeTypes", result);
}

void mlirGoSetTargetDataLayout(MlirModule module, LLVMTargetDataRef layout) {
    auto _module = unwrap(module);
    auto _layout = llvm::unwrap(layout);
    _module->setAttr("llvm.data_layout",
                     ::mlir::StringAttr::get(_module->getContext(), _layout->getStringRepresentation()));

    auto ctx = _module->getContext();

    // Also import this data layout and apply it to the module
    auto spec = mlir::translateDataLayout(*_layout, _module->getContext());
    auto entries = spec.getEntries().vec();

    // Set the index type to be the same width as a pointer in the target machine.
    entries.emplace_back(mlir::DataLayoutEntryAttr::get(
        mlir::IndexType::get(ctx),
        mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), _layout->getPointerSizeInBits())));

    spec = mlir::DataLayoutSpecAttr::get(ctx, entries);

    _module->setAttr(mlir::DLTIDialect::kDataLayoutAttrName, spec);
}

void mlirGoSetTargetTriple(MlirModule module, MlirStringRef triple) {
    auto _module = unwrap(module);
    auto _triple = unwrap(triple);
    _module->setAttr("llvm.target_triple", ::mlir::StringAttr::get(_module->getContext(), _triple));
}

MlirLogicalResult mlirCanonicalizeModule(MlirModule module) {
    auto _module = unwrap(module);
    auto pm = mlir::PassManager::on<mlir::ModuleOp>(_module->getContext());
    pm.addPass(mlir::createCanonicalizerPass());
    return wrap(pm.run(_module));
}

MlirLogicalResult mlirGoOptimizeModule(MlirModule module, MlirStringRef name, MlirStringRef outputDir, bool debug) {
    auto _module = unwrap(module);
    llvm::Twine _name = unwrap(name);
    auto _outputDir = unwrap(outputDir);
    auto pm = mlir::PassManager::on<mlir::ModuleOp>(_module->getContext());
    pm.enableVerifier(true);

    if (debug) {
        pm.getContext()->disableMultithreading();
        pm.enableIRPrinting(IRPrinterConfig::New(_outputDir));
        pm.enableStatistics();
        pm.enableTiming();
    }

    pm.addNestedPass<mlir::go::FuncOp>(mlir::go::createOptimizeDefersPass());
    pm.addPass(mlir::go::createCallPass());
    pm.addPass(mlir::go::createAttachDebugInfoPass());
    pm.addPass(mlir::go::createGlobalConstantsPass());
    pm.addPass(mlir::go::createGlobalInitializerPass());
    pm.addNestedPass<mlir::go::FuncOp>(mlir::go::createHeapEscapePass());

    // Run the canonicalizer pass after Go-centric passes so no context is lost.
    pm.addPass(mlir::createCanonicalizerPass());

    pm.addPass(mlir::go::createLowerToCorePass());
    pm.addPass(mlir::go::createLowerToLLVMPass());
    pm.addNestedPass<mlir::LLVM::LLVMFuncOp>(mlir::LLVM::createLegalizeForExportPass());
    pm.addPass(mlir::createCanonicalizerPass());
    // pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createSymbolDCEPass());
    // pm.addPass(mlir::createSCCPPass());
    pm.addPass(mlir::createCanonicalizerPass());

    return wrap(pm.run(_module));
}

MlirAttribute mlirGetLLVMLinkageAttr(MlirContext context, MlirStringRef linkage) {
    auto _context = unwrap(context);
    auto _linkage = unwrap(linkage);
    if (auto value = mlir::LLVM::linkage::symbolizeLinkage(_linkage); value) {
        return wrap(mlir::LLVM::LinkageAttr::get(_context, *value));
    }
    assert(false && "unreachable");
}

void mlirInitModuleTranslation(MlirContext context) {
    auto _context = unwrap(context);

    // Register the translation to LLVM IR with the MLIR context.
    mlir::registerBuiltinDialectTranslation(*_context);
    mlir::registerLLVMDialectTranslation(*_context);
}

LLVMModuleRef mlirTranslateModuleToLLVMIR(MlirModule module, LLVMContextRef llvmContext, MlirStringRef name) {
    auto _module = unwrap(module);
    auto _llvmContext = llvm::unwrap(llvmContext);
    auto _name = unwrap(name);
    return llvm::wrap(translateModuleToLLVMIR(_module, *_llvmContext, _name).release());
}

MlirAttribute mlirGoCreateTypeMetadataEntryAttr(MlirType type, MlirAttribute dict) {
    const auto _type = mlir::TypeAttr::get(unwrap(type));
    const auto _dict = mlir::cast<mlir::DictionaryAttr>(unwrap(dict));
    return wrap(mlir::go::TypeMetadataEntryAttr::get(_type.getContext(), _type, _dict));
}

MlirAttribute mlirGoCreateTypeMetadataDictionaryAttr(MlirContext context, intptr_t nEntries, MlirAttribute *entries) {
    const auto _context = unwrap(context);
    mlir::SmallVector<mlir::Attribute> values;
    (void) unwrapList(nEntries, entries, values);

    mlir::DenseSet<mlir::go::TypeMetadataEntryAttr> _entries;
    for (auto &value: values) {
        _entries.insert(mlir::cast<mlir::go::TypeMetadataEntryAttr>(value));
    }

    const auto _arrayAttr = mlir::go::TypeMetadataEntryArrayAttr::get(
        _context, mlir::SmallVector<mlir::go::TypeMetadataEntryAttr>(_entries.begin(), _entries.end()));
    return wrap(mlir::go::TypeMetadataDictionaryAttr::get(_context, _arrayAttr));
}

MlirBlock mlirRegionGetLastBlock(MlirRegion region) {
    auto _region = unwrap(region);
    if (!_region->empty())
        return wrap(&_region->getBlocks().back());
    return wrap(static_cast<mlir::Block *>(nullptr));
}

MlirLogicalResult mlirVerifyModule(MlirModule module) {
    auto _module = unwrap(module);
    return wrap(mlir::verify(_module, true));
}

MlirAttribute mlirGoCreateComplexNumberAttr(MlirContext context, MlirType type, double real, double imag) {
    auto _context = unwrap(context);
    auto _type = unwrap(type);
    auto floatType = mlir::cast<mlir::ComplexType>(_type).getElementType();
    return wrap(mlir::go::ComplexNumberAttr::get(_context, mlir::FloatAttr::get(floatType, real),
                                                 mlir::FloatAttr::get(floatType, imag)));
}

bool mlirOperationHasNoMemoryEffect(MlirOperation op) {
    auto _op = unwrap(op);
    auto face = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(_op);
    if (!face || !face.hasNoEffect()) {
        return false;
    }
    return true;
}

MlirOperation mlirValueGetDefiningOperation(MlirValue value) {
    auto _value = unwrap(value);
    return wrap(_value.getDefiningOp());
}

MlirBlock mlirBlockCreate2(intptr_t nArgs, MlirType *args, intptr_t nLocations, MlirLocation *locations) {
    assert(nArgs == nLocations);
    return mlirBlockCreate(nArgs, args, locations);
}

MlirAttribute mlirDistinctAttrGet(MlirAttribute attr) {
    auto _attr = unwrap(attr);
    return wrap(mlir::DistinctAttr::create(_attr));
}

void mlirGoBlockDumpTail(MlirBlock block, intptr_t count) {
    auto _block = unwrap(block);
    auto it = _block->rbegin();
    while (count > 0) {
        it->dump();
        --count;
        ++it;
        if (it == _block->rend()) {
            break;
        }
    }
}
