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
	CGO_LDFLAGS += -fuse-ld=lld -lrt -ldl -lpthread -lm -lz -ltinfo
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

GOFLAGS +=

SIGO_BUILD_RELEASE ?= 0
ifeq ($(SIGO_BUILD_RELEASE),0)
	CGO_LDFLAGS += -g
	CGO_CFLAGS += -O0 -g
	CMAKE_BUILD_TYPE := Debug
else
	CGO_CFLAGS += -Oz
	CMAKE_BUILD_TYPE := Release
endif

LLVM_BUILD_DIR=$(ROOT_DIR)/build/$(CMAKE_BUILD_TYPE)/llvm-build
LLVM_CMAKE_CACHE=$(LLVM_BUILD_DIR)/CMakeCache.txt
LLVM_CONFIG_EXECUTABLE=${LLVM_BUILD_DIR}/bin/llvm-config$(EXECUTABLE_POSTFIX)
LLVM_BUILD_COMPONENTS := ARM AVR RISCV
LLVM_COMPONENTS := ARM AVR RISCV passes

GOIR_ROOT=$(ROOT_DIR)/goir
GOIR_BUILD_DIR=$(ROOT_DIR)/build/$(CMAKE_BUILD_TYPE)/goir-build
GOIR_CMAKE_CACHE=$(GOIR_BUILD_DIR)/CMakeCache.txt
GOIR_LIB=$(GOIR_BUILD_DIR)/libGoIR.a

# Build a semicolon separated list that CMake can accept
CMAKE_LLVM_COMPONENTS :=
$(foreach item, $(LLVM_BUILD_COMPONENTS),$(if $(CMAKE_LLVM_COMPONENTS),$(eval CMAKE_LLVM_COMPONENTS := $(CMAKE_LLVM_COMPONENTS);))$(eval CMAKE_LLVM_COMPONENTS := $(CMAKE_LLVM_COMPONENTS)$(strip $(item))))

# Determine build flags required by LLVM
CGO_LDFLAGS += -Wl,--gc-sections $(shell ${LLVM_CONFIG_EXECUTABLE} --ldflags) $(shell ${LLVM_CONFIG_EXECUTABLE} --libs ${LLVM_COMPONENTS}) -L${GOIR_BUILD_DIR}/lib
CGO_CFLAGS += -fPIC -ffunction-sections -fdata-sections $(shell ${LLVM_CONFIG_EXECUTABLE} --cflags)

# Add MLIR libraries
CGO_LDFLAGS += @link.rsp
CGO_LDFLAGS += -lGoIR -lCGoIR
CGO_LDFLAGS += -lstdc++

# Add MLIR includes
CGO_CFLAGS += -I$(ROOT_DIR)/thirdparty/llvm-project/mlir/include
CGO_CFLAGS += -I${LLVM_BUILD_DIR}/tools/mlir/include
CGO_CFLAGS += -I${GOIR_ROOT}/include
CGO_CFLAGS += -I${GOIR_BUILD_DIR}/include

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

TARGETS_DEVICE_SRCS += $(wildcard $(ROOT_DIR)/targets/device/*.go)
TARGETS_DEVICE_SRCS += $(wildcard $(ROOT_DIR)/targets/device/importer/*.go)
TARGETS_DEVICE_SRCS += $(wildcard $(ROOT_DIR)/targets/device/importer/atdf/*.go)
TARGETS_DEVICE_SRCS += $(wildcard $(ROOT_DIR)/targets/device/importer/svd/*.go)
TARGETS_DEVICE_SRCS += $(wildcard $(ROOT_DIR)/targets/device/svd/*.go)

SSA_TEST_EXE=$(BINDIR)/ssa_test$(EXECUTABLE_POSTFIX)
ABS_SSA_TEST_EXE=$(ABS_BINDIR)/ssa_test$(EXECUTABLE_POSTFIX)

# Common commandline options:
DEBUG ?= 0

define build-compiler-rt
	CC=${CC} CXX=${CXX} cmake $(ROOT_DIR)/thirdparty/llvm-project/compiler-rt -G "Ninja" -B ./build/Release/compiler-rt-$(1)-$(2) \
		-DCMAKE_INSTALL_PREFIX=$(ROOT_DIR)/lib/compiler-rt/$(1)/$(2) \
		${CMAKE_COMPILER_ARGS} \
		-DCMAKE_BUILD_TYPE=Release \
		-DBUILD_SHARED_LIBS=OFF \
		-DCMAKE_SYSTEM_NAME="Generic" \
		-DCMAKE_C_COMPILER_TARGET=$(1) \
		-DCMAKE_C_FLAGS="-nostdlib -march=$(2) $(3)" \
		-DCMAKE_CXX_COMPILER_TARGET=$(1) \
		-DCMAKE_CXX_FLAGS="-nostdlib -march=$(2) $(3)" \
		-DCMAKE_ASM_COMPILER_TARGET=$(1) \
		-DCMAKE_ASM_FLAGS="-march=$(2) $(3)" \
		-DLLVM_CMAKE_DIR=$(LLVM_BUILD_DIR) \
		-DCOMPILER_RT_OS_DIR="$(1)" \
		-DCOMPILER_RT_DEFAULT_TARGET_ONLY=ON \
		-DCOMPILER_RT_BAREMETAL_BUILD=ON \
		-DCOMPILER_RT_BUILD_BUILTINS=ON \
		-DCOMPILER_RT_BUILD_CRT=ON \
		-DCOMPILER_RT_BUILD_SANITIZERS=OFF \
		-DCOMPILER_RT_BUILD_XRAY=OFF \
		-DCOMPILER_RT_BUILD_LIBFUZZER=OFF \
		-DCOMPILER_RT_BUILD_PROFILE=OFF
	cmake --build ./build/Release/compiler-rt-$(1)-$(2) --target install
endef

define build-picolibc
	CC=${CC} CXX=${CXX} cmake $(ROOT_DIR)/thirdparty/picolibc -G "Ninja" -B ./build/Release/picolibc-$(1)-$(2) 	\
		-DCMAKE_INSTALL_PREFIX=$(ROOT_DIR)/lib/picolibc/$(1)/$(2) \
		${CMAKE_COMPILER_ARGS} \
		-DCMAKE_BUILD_TYPE=Release \
		-DBUILD_SHARED_LIBS=OFF \
		-DCMAKE_SYSTEM_NAME="Generic" \
		-DCMAKE_SYSTEM_PROCESSOR="arm" \
		-DCMAKE_C_COMPILER_TARGET=$(1) \
		-DCMAKE_C_FLAGS="-nostdlib -march=$(2) $(3)" \
		-DCMAKE_CXX_COMPILER_TARGET=$(1) \
		-DCMAKE_CXX_FLAGS="-nostdlib -march=$(2) $(3)" \
		-DCMAKE_ASM_COMPILER_TARGET=$(1) \
		-DCMAKE_ASM_FLAGS="-march=$(2) $(3)" \
		-DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
		-DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
		-DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
		-DPICOLIBC_TLS=OFF
	cmake --build ./build/Release/picolibc-$(1)-$(2) --target install --parallel
endef

define build-test
	@rm -f $(1)$(EXECUTABLE_POSTFIX)
	CGO_CFLAGS="$(CGO_CFLAGS)" CGO_LDFLAGS="$(CGO_LDFLAGS) -lstdc++" go test -gcflags "all=-N -l" -ldflags="-linkmode external -extldflags=-Wl,--allow-multiple-definition" -c -o $(1) $(2)
endef

define run-test
	CGO_CFLAGS="$(CGO_CFLAGS)" CGO_LDFLAGS="$(CGO_LDFLAGS) -lstdc++" go test -v -gcflags "all=-N -l" -ldflags="-linkmode external -extldflags=-Wl,--allow-multiple-definition" $(1) -args ${args}
endef

.PHONY: all build-compiler-rt build-goir build-llvm build-mlir build-picolibc build-tests clean clean-clang-bindings clean-llvm-bindings clean-mlir-bindings clean-tests clean-sigo configure-goir configure-llvm configure-mlir debug generate-clang-bindings generate-csp generate-llvm-bindings generate-mlir-bindings sigo ssa_test

all: sigo

clean: clean-sigo clean-llvm-bindings clean-clang-bindings clean-mlir-bindings clean-tests

$(SIGO_EXE): build-goir generate-llvm-bindings generate-mlir-bindings $(GO_SRCS) $(LIBS)
	rm -f $(SIGO_EXE)
	@if [ $(SIGO_BUILD_RELEASE) -eq 1 ]; then \
  		GOFLAGS="$(GOFLAGS)" CGO_CFLAGS="$(CGO_CFLAGS)" CGO_LDFLAGS="$(CGO_LDFLAGS)" go build -o $(SIGO_EXE) -ldflags="-linkmode external" $(ROOT_DIR)/cmd/sigoc; \
  	else \
		GOFLAGS="$(GOFLAGS)" CGO_CFLAGS="$(CGO_CFLAGS)" CGO_LDFLAGS="$(CGO_LDFLAGS)" go build -o $(SIGO_EXE) -gcflags "all=-N -l" -ldflags="-linkmode external" $(ROOT_DIR)/cmd/sigoc; \
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
        -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}" \
        -DCMAKE_CXX_STANDARD_LIBRARIES="${CMAKE_CXX_STANDARD_LIBRARIES}" \
		-DLLVM_ENABLE_PROJECTS="llvm;mlir" \
		-DLLVM_ENABLE_ASSERTIONS=ON \
		-DLLVM_ENABLE_EXPENSIVE_CHECKS=ON \
		-DLLVM_ENABLE_BACKTRACES=ON \
		-DLLVM_TARGETS_TO_BUILD="${CMAKE_LLVM_COMPONENTS}" \
		-DMLIR_INCLUDE_TESTS=OFF \
		-DLLVM_INCLUDE_TESTS=OFF \
		-DCOMPILER_RT_INCLUDE_TESTS=OFF \
		-DCLANG_INCLUDE_TESTS=OFF

configure-llvm: $(LLVM_CMAKE_CACHE)

build-llvm: configure-llvm
	cmake --build ${LLVM_BUILD_DIR} -j$(NUM_JOBS)

$(GOIR_CMAKE_CACHE):
	CC=${CC} CXX=${CXX} LD=${LD}  cmake -G "Ninja" -B ${GOIR_BUILD_DIR} ${GOIR_ROOT} \
		-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		${CMAKE_COMPILER_ARGS} \
		${CMAKE_COMPILER_TARGET_ARGS} \
		${CMAKE_LINKER_ARGS} \
		-DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}" \
		-DCMAKE_CXX_STANDARD_LIBRARIES="${CMAKE_CXX_STANDARD_LIBRARIES}" \
		-DCMAKE_PREFIX_PATH=${LLVM_BUILD_DIR}/lib/cmake

configure-goir: build-llvm $(GOIR_CMAKE_CACHE)

$(GOIR_LIB): configure-goir
	cmake --build ${GOIR_BUILD_DIR} -j$(NUM_JOBS)

build-goir: $(GOIR_LIB) ./mlir/mlir.go

configure: configure-llvm configure-goir

reconfigure:
	rm -f $(LLVM_CMAKE_CACHE) $(GOIR_CMAKE_CACHE)
	"$(MAKE)" configure

generate-llvm-bindings: ./llvm/llvm.go
./llvm/llvm.go: ./llvm/llvm.i
	@echo "Generating LLVM bindings using SWIG..."
	@swig -DSWIGWORDSIZE64 -go -intgosize 64 -cgo \
	-I$(ROOT_DIR)/build/llvm-build/lib/clang/16/include \
	-I$(ROOT_DIR)/build/llvm-build/include \
	-I$(ROOT_DIR)/build/llvm-headers/include \
	-I$(ROOT_DIR)/build/llvm-build/include \
	-I$(ROOT_DIR)/thirdparty/llvm-project/llvm/include \
	$(ROOT_DIR)/llvm/llvm.i
	@echo "Done."

clean-llvm-bindings:
	rm $(ROOT_DIR)/llvm/llvm.go \
	   $(ROOT_DIR)/llvm/llvm_wrap.c

generate-clang-bindings: ./clang/clang.go
./clang/clang.go: ./clang/clang.i
	@echo "Generating Clang bindings using SWIG..."
	@swig -DSWIGWORDSIZE64 -go -intgosize 64 -cgo \
	-I$(ROOT_DIR)/build/llvm-build/lib/clang/16/include \
	-I$(ROOT_DIR)/build/llvm-build/include \
	-I$(ROOT_DIR)/build/llvm-build/include \
	-I$(ROOT_DIR)/thirdparty/llvm-project/clang/include \
	$(ROOT_DIR)/clang/clang.i
	@echo "Done."

clean-clang-bindings:
	rm $(ROOT_DIR)/clang/clang.go \
	   $(ROOT_DIR)/clang/clang_wrap.c

generate-mlir-bindings: ./mlir/mlir.go
./mlir/mlir.go: ./mlir/mlir.i $(GOIR_ROOT)/include/Go-c/mlir/Dialects.h $(GOIR_ROOT)/include/Go-c/mlir/Enums.h $(GOIR_ROOT)/include/Go-c/mlir/Operations.h $(GOIR_ROOT)/include/Go-c/mlir/Types.h
	@echo "Generating MLIR bindings using SWIG..."
	@swig -DSWIGWORDSIZE64 -go -intgosize 64 -cgo \
	-I$(ROOT_DIR)/build/llvm-build/lib/clang/16/include \
	-I$(ROOT_DIR)/build/llvm-build/include \
	-I${GOIR_ROOT}/include \
	-I$(ROOT_DIR)/thirdparty/llvm-project/mlir/include \
	-I$(ROOT_DIR)/thirdparty/llvm-project/llvm/include \
	$(ROOT_DIR)/mlir/mlir.i
	@echo "Done."

clean-mlir-bindings:
	rm $(ROOT_DIR)/mlir/mlir.go \
	   $(ROOT_DIR)/mlir/mlir_wrap.c

build-picolibc:
	$(call build-picolibc,armv7m-none-eabi,armv7m+fp,-mthumb)
	$(call build-picolibc,armv7m-none-eabi,armv7m+nofp,-mthumb)
	$(call build-picolibc,armv7em-none-eabi,armv7em+fp,-mthumb)
	$(call build-picolibc,armv7em-none-eabi,armv7em+nofp,-mthumb)
	$(call build-picolibc,armv6m-none-eabi,armv6m+nofp,-mthumb)

build-compiler-rt:
	$(call build-compiler-rt,armv7m-none-eabi,armv7m+fp,-mthumb)
	$(call build-compiler-rt,armv7m-none-eabi,armv7m+nofp,-mthumb)
	$(call build-compiler-rt,armv7em-none-eabi,armv7em+fp,-mthumb)
	$(call build-compiler-rt,armv7em-none-eabi,armv7em+nofp,-mthumb)
	$(call build-compiler-rt,armv6m-none-eabi,armv6m+nofp,-mthumb)

$(CSP_GEN_EXE): $(wildcard $(ROOT_DIR)/cmd/csp-gen/*.go)
	@if [ $(SIGO_BUILD_RELEASE) -eq 1 ]; then \
		go build -o $(CSP_GEN_EXE) -gcflags "all=-N -l" $(ROOT_DIR)/cmd/csp-gen; \
  	else \
		go build -o $(CSP_GEN_EXE) -gcflags "all=-N -l" $(ROOT_DIR)/cmd/csp-gen; \
	fi

csp-gen: $(CSP_GEN_EXE)
	@if [ $(DEBUG) -eq 1 ]; then \
		dlv --listen=:2346 --headless=true --api-version=2 --accept-multiclient exec $(CSP_GEN_EXE) -- $(args); \
	else \
		$(CSP_GEN_EXE) $(args); \
	fi

generate-csp: $(CSP_GEN_EXE)
	$(CSP_GEN_EXE) --in=$(ROOT_DIR)/targets/definitions/cortexm.json --out=$(ROOT_DIR)/src/runtime/arm/cortexm/support
	$(CSP_GEN_EXE) --in=$(ROOT_DIR)/targets/definitions/atsamd21.json --out=$(ROOT_DIR)/src/runtime/arm/cortexm/sam/atsamd21/support
	$(CSP_GEN_EXE) --in=$(ROOT_DIR)/targets/definitions/atsamx5x.json --out=$(ROOT_DIR)/src/runtime/arm/cortexm/sam/atsamx5x/support
	$(CSP_GEN_EXE) --in=$(ROOT_DIR)/targets/definitions/stm32h747_cm7.json --out=$(ROOT_DIR)/src/runtime/arm/cortexm/stm32/stm32h7x7/support

$(DEF_GEN_EXE): $(wildcard $(ROOT_DIR)/cmd/def-gen/*.go) $(TARGETS_DEVICE_SRCS)
	@if [ $(SIGO_BUILD_RELEASE) -eq 1 ]; then \
		go build -o $(DEF_GEN_EXE) $(ROOT_DIR)/cmd/def-gen; \
  	else \
		go build -o $(DEF_GEN_EXE) -gcflags "all=-N -l" $(ROOT_DIR)/cmd/def-gen; \
	fi
def-gen: $(DEF_GEN_EXE)
	@if [ $(DEBUG) -eq 1 ]; then \
		dlv --listen=:2346 --headless=true --api-version=2 --accept-multiclient exec $(DEF_GEN_EXE) -- $(args); \
	else \
		$(DEF_GEN_EXE) $(args); \
	fi

release: build-picolibc build-compiler-rt generate-csp sigo
