ROOT_DIR := $(patsubst %/,%,$(dir $(realpath $(lastword $(MAKEFILE_LIST)))))

ifeq ($(OS),Windows_NT)
	EXECUTABLE_POSTFIX=.exe
	CGO_LDFLAGS += -lole32 -luuid -lpsapi -lshell32 -ladvapi32 -lntdll
	CMAKE_CXX_FLAGS += -pthread -femulated-tls
	CMAKE_CXX_STANDARD_LIBRARIES += -lpthread
	CLANG_TARGET := x86_64-pc-windows-gnu
	CC ?= clang
	CXX ?= clang++
	# NOTE: ld should be replaced with ld.lld directly on Windows since Go is dumb.
else
	CGO_LDFLAGS += -fuse-ld=lld -lrt -ldl -lpthread -lm -lz -ltinfo -lzstd
endif

CMAKE_COMPILER_ARGS := -DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_COMPILER=${CXX}
CMAKE_COMPILER_TARGET_ARGS += -DCMAKE_C_COMPILER_TARGET=${CLANG_TARGET} -DCMAKE_CXX_COMPILER_TARGET=${CLANG_TARGET}
CMAKE_LINKER_ARGS := -DCMAKE_LINKER_TYPE=DEFAULT

# Override the linker based on pattern
ifneq (,$(findstring ld.gold,$(LD)))
	CMAKE_LINKER_ARGS := -DCMAKE_LINKER_TYPE=GOLD
else ifneq (,$(findstring ld.lld,$(LD)))
	CMAKE_LINKER_ARGS := -DCMAKE_LINKER_TYPE=LLD
endif

SIGO_BUILD_RELEASE ?= 0
ifeq ($(SIGO_BUILD_RELEASE),0)
	CGO_LDFLAGS += -g
	CGO_CFLAGS += -O0 -g
	CMAKE_BUILD_TYPE := Debug
else
	CGO_CFLAGS += -Oz
	CMAKE_BUILD_TYPE := Release
endif

LLVM_SRC_DIR=$(ROOT_DIR)/thirdparty/llvm-project
LLVM_BUILD_DIR=$(ROOT_DIR)/build/$(CMAKE_BUILD_TYPE)/llvm-build
LLVM_CMAKE_CACHE=$(LLVM_BUILD_DIR)/CMakeCache.txt
LLVM_CONFIG_EXECUTABLE=${LLVM_BUILD_DIR}/bin/llvm-config$(EXECUTABLE_POSTFIX)
LLVM_BUILD_COMPONENTS := ARM AVR RISCV
LLVM_COMPONENTS := ARM AVR RISCV passes

GOIR_ROOT=$(ROOT_DIR)/goir
GOIR_BUILD_DIR=$(ROOT_DIR)/build/$(CMAKE_BUILD_TYPE)/goir-build
GOIR_CMAKE_CACHE=$(GOIR_BUILD_DIR)/CMakeCache.txt
GOIR_LIB=$(GOIR_BUILD_DIR)/libGoIR.a

CLANG_ROOT=$(ROOT_DIR)/clang
CLANG_BUILD_DIR=$(ROOT_DIR)/build/$(CMAKE_BUILD_TYPE)/clang-build
CLANG_CMAKE_CACHE=$(CLANG_BUILD_DIR)/CMakeCache.txt
CLANG_LIB=$(CLANG_BUILD_DIR)/lib/libCGoClang.a

INSTALL_DIR=$(ROOT_DIR)/build/$(CMAKE_BUILD_TYPE)/install

# Build a semicolon separated list that CMake can accept
CMAKE_LLVM_COMPONENTS :=
$(foreach item, $(LLVM_BUILD_COMPONENTS),$(if $(CMAKE_LLVM_COMPONENTS),$(eval CMAKE_LLVM_COMPONENTS := $(CMAKE_LLVM_COMPONENTS);))$(eval CMAKE_LLVM_COMPONENTS := $(CMAKE_LLVM_COMPONENTS)$(strip $(item))))

# Determine build flags required by LLVM
CGO_LDFLAGS += -Wl,--gc-sections $(shell ${LLVM_CONFIG_EXECUTABLE} --ldflags) $(shell ${LLVM_CONFIG_EXECUTABLE} --libs ${LLVM_COMPONENTS}) -L${GOIR_BUILD_DIR}/lib
CGO_LDFLAGS += -lLLVMTableGen -lLLVMOption -lLLVMPlugins
CGO_CFLAGS += -fPIC -ffunction-sections -fdata-sections $(shell ${LLVM_CONFIG_EXECUTABLE} --cflags)

# Add MLIR libraries
#CGO_LDFLAGS += @link.rsp
CGO_LDFLAGS += -lGoIR -lCGoIR
CGO_LDFLAGS += -lstdc++

# Add clang support libraries
CGO_LDFLAGS += -L${CLANG_BUILD_DIR}/lib -L${CLANG_BUILD_DIR}/lib/CAPI -lGoClangSupport -lCGoClang
CGO_LDFLAGS += -lclangAnalysis -lclangAnalysisFlowSensitive -lclangAnalysisFlowSensitiveModels -lclangAnalysisLifetimeSafety -lclangAnalysisScalable -lclangAPINotes -lclangAST -lclangASTMatchers -lclangBasic -lclangCIR -lclangCIRFrontendAction -lclangCIRLoweringCommon -lclangCIRLoweringDirectToLLVM -lclangCodeGen -lclangCrossTU -lclangDependencyScanning -lclangDirectoryWatcher -lclangDriver -lclangDynamicASTMatchers -lclangEdit -lclangExtractAPI -lclangFormat -lclangFrontend -lclangFrontendTool -lclangHandleCXX -lclangHandleLLVM -lclangIndex -lclangIndexSerialization -lclangInstallAPI -lclangInterpreter -lclangLex -lclangOptions -lclangParse -lclangRewrite -lclangRewriteFrontend -lclangSema -lclangSerialization -lclangStaticAnalyzerCheckers -lclangStaticAnalyzerCore -lclangStaticAnalyzerFrontend -lclangSupport -lclangTooling -lclangToolingASTDiff -lclangToolingCore -lclangToolingInclusions -lclangToolingInclusionsStdlib -lclangToolingRefactoring -lclangToolingSyntax -lclangTransformer
CGO_LDFLAGS += -lMLIRCIR -lMLIRCIRInterfaces -lMLIRCIRTargetLowering -lMLIRCIRTransforms -lCIROpenACCSupport

# Add LLVM includes
CGO_CFLAGS += -I$(ROOT_DIR)/thirdparty/llvm-project/llvm/include
CGO_CFLAGS += -I${LLVM_BUILD_DIR}/tools/mlir/include

# Add MLIR includes
CGO_CFLAGS += -I$(ROOT_DIR)/thirdparty/llvm-project/mlir/include
CGO_CFLAGS += -I${GOIR_ROOT}/include
CGO_CFLAGS += -I${GOIR_BUILD_DIR}/include

# Add clang support includes
CGO_CFLAGS += -I${CLANG_ROOT}/include
CGO_CFLAGS += -I$(ROOT_DIR)/thirdparty/llvm-project/clang/include
CGO_CFLAGS += -I${LLVM_BUILD_DIR}/tools/clang/include

CGO_CXXFLAGS := -std=c++17 -fno-rtti $(CGO_CFLAGS)

# Paths:
BINDIR := ./bin
ABS_BINDIR := $(ROOT_DIR)/bin

# Sources:
GO_BUILDER_SRCS := $(wildcard $(ROOT_DIR)/builder/*.go)
GO_COMPILER_SSA_SRCS := $(wildcard $(ROOT_DIR)/compiler/ssa/*.go)
GO_CMD_SIGOC_SRCS :=  $(wildcard $(ROOT_DIR)/cmd/sigoc/*.go)
GO_LLVM_SRCS :=  $(wildcard $(ROOT_DIR)/llvm/*.go)
GO_MLIR_SRCS :=  $(wildcard $(ROOT_DIR)/mlir/*.go)
GO_SRCS := $(GO_BUILDER_SRCS) $(GO_COMPILER_SSA_SRCS) $(GO_CMD_SIGOC_SRCS) $(GO_LLVM_SRCS) $(GO_MLIR_SRCS)

# Libraries:
LIBS := $(wildcard $(GOIR_BUILD_DIR)/lib/*.a) $(wildcard $(LLVM_BUILD_DIR)/lib/*.a)

# Executables:
SIGO_EXE=$(BINDIR)/sigoc$(EXECUTABLE_POSTFIX)
ABS_SIGO_EXE=$(ABS_BINDIR)/sigoc$(EXECUTABLE_POSTFIX)

CSP_GEN_EXE=$(BINDIR)/csp-gen$(EXECUTABLE_POSTFIX)
DEF_GEN_EXE=$(BINDIR)/def-gen$(EXECUTABLE_POSTFIX)
TBDEF_GEN_EXE=$(BINDIR)/tbdef-gen$(EXECUTABLE_POSTFIX)
TABLEGEN_CSP_EXE := $(BINDIR)/tablegen-csp$(EXECUTABLE_POSTFIX)
BIN2STR_EXE := $(BINDIR)/bin2str$(EXECUTABLE_POSTFIX)

TARGETS_DEVICE_SRCS += $(wildcard $(ROOT_DIR)/targets/device/*.go)
TARGETS_DEVICE_SRCS += $(wildcard $(ROOT_DIR)/targets/device/importer/*.go)
TARGETS_DEVICE_SRCS += $(wildcard $(ROOT_DIR)/targets/device/importer/atdf/*.go)
TARGETS_DEVICE_SRCS += $(wildcard $(ROOT_DIR)/targets/device/importer/svd/*.go)
TARGETS_DEVICE_SRCS += $(wildcard $(ROOT_DIR)/targets/device/svd/*.go)

SSA_TEST_EXE=$(BINDIR)/ssa_test$(EXECUTABLE_POSTFIX)
ABS_SSA_TEST_EXE=$(ABS_BINDIR)/ssa_test$(EXECUTABLE_POSTFIX)

# Common commandline options:
DEBUG ?= 0

define build-test
	@rm -f $(1)$(EXECUTABLE_POSTFIX)
	CGO_CFLAGS="$(CGO_CFLAGS)" CGO_LDFLAGS="$(CGO_LDFLAGS) -lstdc++" go test -gcflags "all=-N -l" -ldflags="-linkmode external -extldflags=-Wl,--allow-multiple-definition" -c -o $(1) $(2)
endef

define run-test
	CGO_CFLAGS="$(CGO_CFLAGS)" CGO_LDFLAGS="$(CGO_LDFLAGS) -lstdc++" go test -v -gcflags "all=-N -l" -ldflags="-linkmode external -extldflags=-Wl,--allow-multiple-definition" $(1) -args ${args}
endef

.PHONY: all build-clang build-goir build-llvm build-mlir build-tests clean clean-tests clean-sigo configure-clang configure-goir configure-llvm configure-mlir debug generate-csp sigo ssa_test

all: sigo

env:
	CGO_CFLAGS="$(CGO_CFLAGS)" CGO_LDFLAGS="$(CGO_LDFLAGS)" go env

clean: clean-sigo clean-tests clean-sysroots

$(SIGO_EXE): build-goir build-clang sysroots $(GO_SRCS) $(LIBS)
	rm -f $(SIGO_EXE)
	@if [ $(SIGO_BUILD_RELEASE) -eq 1 ]; then \
  		CGO_CFLAGS="$(CGO_CFLAGS)" CGO_LDFLAGS="$(CGO_LDFLAGS)" CGO_CXXFLAGS="$(CGO_CXXFLAGS)" go build -o $(SIGO_EXE) -ldflags="-linkmode external" $(ROOT_DIR)/cmd/sigoc; \
  	else \
		CGO_CFLAGS="$(CGO_CFLAGS)" CGO_LDFLAGS="$(CGO_LDFLAGS)" CGO_CXXFLAGS="$(CGO_CXXFLAGS)" go build -o $(SIGO_EXE) -gcflags "all=-N -l" -ldflags="-linkmode external" $(ROOT_DIR)/cmd/sigoc; \
  	fi

sigo: $(SIGO_EXE)

debug: sigo
	dlv --listen=:2346 --headless=true --api-version=2 --accept-multiclient exec $(SIGO_EXE) -- ${args}

run-tests:
	$(call run-test,./compiler/ssa)

$(SSA_TEST_EXE): $(GO_COMPILER_SSA_SRCS) $(LIBS)
	$(call build-test, $@, ./compiler/ssa)

ssa_test: $(SSA_TEST_EXE)

build-tests: ssa_test

test: build-tests
	@if [ $(DEBUG) -eq 1 ]; then \
		dlv --listen=:2346 --headless=true --api-version=2 --accept-multiclient exec --wd=$(ROOT_DIR)/compiler/ssa $(ABS_SSA_TEST_EXE) -- -tests=${TESTS}; \
	else \
		cd ./compiler/ssa && $(ABS_SSA_TEST_EXE) -tests=${TESTS}; \
	fi

clean-tests:
	rm $(SSA_TEST_EXE)

clean-sigo:
	rm $(SIGO_EXE)

$(LLVM_CMAKE_CACHE):
	@mkdir -p ${LLVM_BUILD_DIR}
	CC=${CC} CXX=${CXX} LD=${LD} cmake -G "Ninja" -B ${LLVM_BUILD_DIR} $(ROOT_DIR)/thirdparty/llvm-project/llvm 	\
		-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		${CMAKE_COMPILER_ARGS} \
		${CMAKE_COMPILER_TARGET_ARGS} \
		${CMAKE_LINKER_ARGS} \
		-DCMAKE_INSTALL_PREFIX=$(INSTALL_DIR) \
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}" \
    -DCMAKE_CXX_STANDARD_LIBRARIES="${CMAKE_CXX_STANDARD_LIBRARIES}" \
		-DLLVM_ENABLE_PROJECTS="clang;llvm;mlir" \
		-DLLVM_ENABLE_ASSERTIONS=ON \
		-DLLVM_ENABLE_EXPENSIVE_CHECKS=ON \
		-DLLVM_ENABLE_BACKTRACES=ON \
		-DLLVM_TARGETS_TO_BUILD="${CMAKE_LLVM_COMPONENTS}" \
		-DMLIR_INCLUDE_TESTS=OFF \
		-DLLVM_INCLUDE_TESTS=OFF \
		-DCOMPILER_RT_INCLUDE_TESTS=OFF \
		-DCLANG_INCLUDE_TESTS=OFF \
		-DCLANG_ENABLE_CIR=ON

configure-llvm: $(LLVM_CMAKE_CACHE)

build-llvm: configure-llvm
	cmake --build ${LLVM_BUILD_DIR} -j$(NUM_JOBS)

install-llvm: build-llvm
	cmake --install ${LLVM_BUILD_DIR} --prefix $(INSTALL_DIR)

$(GOIR_CMAKE_CACHE):
	CC=${CC} CXX=${CXX} LD=${LD}  cmake -G "Ninja" -B ${GOIR_BUILD_DIR} ${GOIR_ROOT} \
		-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		${CMAKE_COMPILER_ARGS} \
		${CMAKE_COMPILER_TARGET_ARGS} \
		${CMAKE_LINKER_ARGS} \
		-DCMAKE_INSTALL_PREFIX=$(INSTALL_DIR) \
		-DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}" \
		-DCMAKE_CXX_STANDARD_LIBRARIES="${CMAKE_CXX_STANDARD_LIBRARIES}" \
		-DCMAKE_PREFIX_PATH=${LLVM_BUILD_DIR}/lib/cmake

configure-goir: build-llvm $(GOIR_CMAKE_CACHE)

$(GOIR_LIB): configure-goir
	cmake --build ${GOIR_BUILD_DIR} -j$(NUM_JOBS)

build-goir: $(GOIR_LIB)

install-goir: build-goir
	cmake --install ${GOIR_BUILD_DIR} --prefix $(INSTALL_DIR)

$(CLANG_CMAKE_CACHE):
	CC=${CC} CXX=${CXX} LD=${LD} cmake -G "Ninja" -B ${CLANG_BUILD_DIR} ${CLANG_ROOT} \
		-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		${CMAKE_COMPILER_ARGS} \
		${CMAKE_COMPILER_TARGET_ARGS} \
		${CMAKE_LINKER_ARGS} \
		-DCMAKE_INSTALL_PREFIX=$(INSTALL_DIR) \
		-DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}" \
		-DCMAKE_CXX_STANDARD_LIBRARIES="${CMAKE_CXX_STANDARD_LIBRARIES}" \
		-DCMAKE_PREFIX_PATH=${LLVM_BUILD_DIR}/lib/cmake

configure-clang: build-llvm $(CLANG_CMAKE_CACHE)

$(CLANG_LIB): configure-clang
	cmake --build ${CLANG_BUILD_DIR} -j$(NUM_JOBS)

build-clang: $(CLANG_LIB)

configure: configure-llvm configure-goir configure-clang

reconfigure:
	rm -f $(LLVM_CMAKE_CACHE) $(GOIR_CMAKE_CACHE) $(CLANG_CMAKE_CACHE)
	"$(MAKE)" configure

$(TBDEF_GEN_EXE): $(wildcard $(ROOT_DIR)/cmd/tbdef-gen/*.go) $(TARGETS_DEVICE_SRCS)
	@if [ $(SIGO_BUILD_RELEASE) -eq 1 ]; then \
		go build -o $(TBDEF_GEN_EXE) $(ROOT_DIR)/cmd/tbdef-gen; \
  	else \
		go build -o $(TBDEF_GEN_EXE) -gcflags "all=-N -l" $(ROOT_DIR)/cmd/tbdef-gen; \
	fi
tbdef-gen: $(TBDEF_GEN_EXE)
	@if [ $(DEBUG) -eq 1 ]; then \
		dlv --listen=:2346 --headless=true --api-version=2 --accept-multiclient exec $(TBDEF_GEN_EXE) -- $(args); \
	fi

$(TABLEGEN_CSP_EXE): generate-llvm-bindings $(GO_SRCS) $(LIBS)
	rm -f $(TABLEGEN_CSP_EXE)
	@if [ $(SIGO_BUILD_RELEASE) -eq 1 ]; then \
  		CGO_CFLAGS="$(CGO_CFLAGS)" CGO_LDFLAGS="$(CGO_LDFLAGS)" CGO_CXXFLAGS="$(CGO_CXXFLAGS)" go build -o $(TABLEGEN_CSP_EXE) -ldflags="-linkmode external" $(ROOT_DIR)/cmd/tablegen-csp; \
  	else \
		CGO_CFLAGS="$(CGO_CFLAGS)" CGO_LDFLAGS="$(CGO_LDFLAGS)" CGO_CXXFLAGS="$(CGO_CXXFLAGS)" go build -o $(TABLEGEN_CSP_EXE) -gcflags "all=-N -l" -ldflags="-linkmode external" $(ROOT_DIR)/cmd/tablegen-csp; \
  	fi

tablegen-csp: $(TABLEGEN_CSP_EXE)
	@if [ $(DEBUG) -eq 1 ]; then \
    	dlv --listen=:2346 --headless=true --api-version=2 --accept-multiclient exec $(TABLEGEN_CSP_EXE) -- $(args); \
	fi

$(BIN2STR_EXE):
	rm -f $(BIN2STR_EXE)
	@if [ $(SIGO_BUILD_RELEASE) -eq 1 ]; then \
  		go build -o $(BIN2STR_EXE) -ldflags="-linkmode external" $(ROOT_DIR)/cmd/bin2str; \
  	else \
		go build -o $(BIN2STR_EXE) -gcflags "all=-N -l" -ldflags="-linkmode external" $(ROOT_DIR)/cmd/bin2str; \
  	fi

bin2str: $(BIN2STR_EXE)
	@if [ $(DEBUG) -eq 1 ]; then \
    	dlv --listen=:2346 --headless=true --api-version=2 --accept-multiclient exec $(BIN2STR_EXE) -- $(args); \
	fi

release: sigo

include sysroot.mk

