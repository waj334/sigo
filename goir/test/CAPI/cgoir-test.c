// RUN: standalone-capi-test 2>&1 | FileCheck %s

#include <stdio.h>

#include "Go-c/mlir/Dialects.h"
#include <mlir-c/IR.h>
#include <mlir-c/RegisterEverything.h>

static void registerAllUpstreamDialects(MlirContext ctx) {
    MlirDialectRegistry registry = mlirDialectRegistryCreate();
    mlirRegisterAllDialects(registry);
    mlirContextAppendDialectRegistry(ctx, registry);
    mlirDialectRegistryDestroy(registry);
}

int main(int argc, char **argv) {
    MlirContext ctx = mlirContextCreate();
    // TODO: Create the dialect handles for the builtin dialects and avoid this.
    // This adds dozens of MB of binary size over just the standalone dialect.
    registerAllUpstreamDialects(ctx);
    mlirDialectHandleRegisterDialect(mlirGetDialectHandle__go__(), ctx);

    MlirModule module = mlirModuleCreateParse(
            ctx, mlirStringRefCreateFromCString(
                    "%0 = go.alloc {elementType=i32}\n"
                        ));
    if (mlirModuleIsNull(module)) {
        printf("ERROR: Could not parse.\n");
        mlirContextDestroy(ctx);
        return 1;
    }
    MlirOperation op = mlirModuleGetOperation(module);

    // CHECK: %[[C:.*]] = go.alloc i32
    mlirOperationDump(op);

    mlirModuleDestroy(module);
    mlirContextDestroy(ctx);
    return 0;
}