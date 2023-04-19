ROOT_DIR := $(patsubst %/,%,$(dir $(realpath $(lastword $(MAKEFILE_LIST)))))

ifeq ($(OS),Windows_NT)
	EXECUTABLE_POSTFIX=.exe
	CGO_LDFLAGS += -lole32 -luuid -lpsapi -lshell32 -ladvapi32
endif
C_FOR_GO_EXECUTABLE=c-for-go${EXECUTABLE_POSTFIX}
LLVM_BUILD_DIR=${ROOT_DIR}/build/llvm-build
LLVM_CONFIG_EXECUTABLE=${LLVM_BUILD_DIR}/bin/llvm-config${EXECUTABLE_POSTFIX}
LLVM_COMPONENTS := ARM AVR RISCV

# Build a semicolon separated list that CMake can accept
CMAKE_LLVM_COMPONENTS :=
$(foreach item, $(LLVM_COMPONENTS),$(if $(CMAKE_LLVM_COMPONENTS),$(eval CMAKE_LLVM_COMPONENTS := $(CMAKE_LLVM_COMPONENTS);))$(eval CMAKE_LLVM_COMPONENTS := $(CMAKE_LLVM_COMPONENTS)$(strip $(item))))

# Determine build flags required by LLVM
CGO_LDFLAGS += -O2 -g $(shell ${LLVM_CONFIG_EXECUTABLE} --ldflags) $(shell ${LLVM_CONFIG_EXECUTABLE} --libs ${LLVM_COMPONENTS})
CGO_CFLAGS += -O2 -g -fPIC $(shell ${LLVM_CONFIG_EXECUTABLE} --cflags)

SIGO_EXECUTABLE=sigoc${EXECUTABLE_POSTFIX}

# These LLVM header will be preprocessed before handing them to SWIG
LLVM_PREPROCESS_HEADERS += llvm-c/Target.h llvm-c/TargetMachine.h
LLVM_PREPROCESS_HEADERS_OUT := ${ROOT_DIR}/build/llvm-headers/include

define build-compiler-rt
	cmake $(ROOT_DIR)/thirdparty/llvm-project/compiler-rt -G "Ninja" -B ./build/compiler-rt-$(1)-$(2) \
		-DCMAKE_INSTALL_PREFIX=$(ROOT_DIR)/lib/compiler-rt \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_SYSTEM_NAME="Generic" \
		-DCMAKE_C_COMPILER=$(LLVM_BUILD_DIR)/bin/clang$(EXECUTABLE_POSTFIX) \
		-DCMAKE_C_COMPILER_TARGET=$(1) \
		-DCMAKE_C_FLAGS="-nostdlib -march=$(2) $(3)" \
		-DCMAKE_CXX_COMPILER=$(LLVM_BUILD_DIR)/bin/clang++$(EXECUTABLE_POSTFIX) \
		-DCMAKE_CXX_COMPILER_TARGET=$(1) \
		-DCMAKE_CXX_FLAGS="-nostdlib -march=$(2) $(3)" \
		-DCMAKE_ASM_COMPILER_TARGET=$(1) \
		-DCMAKE_ASM_FLAGS="-march=$(2) $(3)" \
		-DCMAKE_LINKER=$(LLVM_BUILD_DIR)/bin/lld$(EXECUTABLE_POSTFIX) \
		-DCMAKE_AR=$(LLVM_BUILD_DIR)/bin/llvm-ar$(EXECUTABLE_POSTFIX) \
		-DCMAKE_NM=$(LLVM_BUILD_DIR)/bin/llvm-nm$(EXECUTABLE_POSTFIX) \
		-DCMAKE_RANLIB=$(LLVM_BUILD_DIR)/bin/llvm-ranlib$(EXECUTABLE_POSTFIX) \
		-DLLVM_CMAKE_DIR=$(LLVM_BUILD_DIR) \
		-DCOMPILER_RT_OS_DIR="$(1)" \
		-DCOMPILER_RT_DEFAULT_TARGET_ONLY=ON \
		-DCOMPILER_RT_BAREMETAL_BUILD=ON \
		-DCOMPILER_RT_BUILD_BUILTINS=ON \
		-DCOMPILER_RT_BUILD_CRT=OFF \
		-DCOMPILER_RT_BUILD_SANITIZERS=OFF \
		-DCOMPILER_RT_BUILD_XRAY=OFF \
		-DCOMPILER_RT_BUILD_LIBFUZZER=OFF \
		-DCOMPILER_RT_BUILD_PROFILE=OFF
	cmake --build ./build/compiler-rt-$(1)-$(2) --target install
endef

.PHONY: all clean sigo clean-sigo configure-llvm build-llvm generate-llvm-bindings clean-llvm-bindings

all: sigo

clean: clean-sigo clean-llvm-bindings

sigo: ./llvm/llvm.go
	CGO_CFLAGS="${CGO_CFLAGS}" CGO_LDFLAGS="${CGO_LDFLAGS} -lstdc++" go build -o ${ROOT_DIR}/bin/${SIGO_EXECUTABLE} -gcflags "all=-N -l" -ldflags="-linkmode external -extldflags=-Wl,--allow-multiple-definition" ${ROOT_DIR}/cmd/sigoc

debug: sigo
	dlv --listen=:2346 --headless=true --api-version=2 --accept-multiclient exec ${ROOT_DIR}/bin/${SIGO_EXECUTABLE} -- ${args}

build-test:
	@rm -f ./bin/test${EXECUTABLE_POSTFIX}
	CGO_CFLAGS="${CGO_CFLAGS}" CGO_LDFLAGS="${CGO_LDFLAGS} -lstdc++" go test -gcflags "all=-N -l" -ldflags="-linkmode external -extldflags=-Wl,--allow-multiple-definition" -c -o ./bin/test${EXECUTABLE_POSTFIX} ${package}

test: build-test
	./bin/test${EXECUTABLE_POSTFIX}

debug-test: build-test
	dlv --listen=:2346 --headless=true --api-version=2 --accept-multiclient exec ./bin/test${EXECUTABLE_POSTFIX}

clean-sigo:
	rm ${ROOT_DIR}/bin/${SIGO_EXECUTABLE}

configure-llvm:
	@CC=gcc
	@CXX=g++
	@mkdir -p ${LLVM_BUILD_DIR}
	cmake -G "Ninja" -B ${LLVM_BUILD_DIR} ${ROOT_DIR}/thirdparty/llvm-project/llvm \
		-DCMAKE_BUILD_TYPE=Release \
		-DLLVM_ENABLE_PROJECTS="clang;llvm;lld" \
		-DLLVM_ENABLE_ASSERTIONS=ON \
		-DLLVM_ENABLE_EXPENSIVE_CHECKS=ON \
		-DLLVM_ENABLE_BACKTRACES=ON \
		-DLLVM_TARGETS_TO_BUILD="${CMAKE_LLVM_COMPONENTS}"

build-llvm: configure-llvm
	cmake --build ${LLVM_BUILD_DIR} -j$(NUM_JOBS)
	make preprocess-llvm-headers

preprocess-llvm-headers:
	$(foreach item,$(LLVM_PREPROCESS_HEADERS),$(call remove-typedefs,$(item)))

define remove-typedefs
	@echo "Preprocessing $(1)"
	@mkdir -p $(dir ${LLVM_PREPROCESS_HEADERS_OUT}/$(1))
	@sed '/typedef struct LLVM/d' ${ROOT_DIR}/thirdparty/llvm-project/llvm/include/$(1) > ${LLVM_PREPROCESS_HEADERS_OUT}/$(1)

endef

generate-llvm-bindings: ./llvm/llvm.go
./llvm/llvm.go: ./llvm/llvm.i ./build/llvm-headers/include/llvm-c
	@echo "Generating LLVM bindings using SWIG..."
	@swig -go -intgosize 64 -cgo \
	-IC:/ProgramData/chocolatey/lib/mingw/tools/install/mingw64/x86_64-w64-mingw32/include \
	-IC:/ProgramData/chocolatey/lib/mingw/tools/install/mingw64/lib/gcc/x86_64-w64-mingw32/12.2.0/include \
	-I${ROOT_DIR}/build/llvm-headers/include \
	-I${ROOT_DIR}/build/llvm-build/include \
	-I${ROOT_DIR}/thirdparty/llvm-project/llvm/include \
	${ROOT_DIR}/llvm/llvm.i
	@echo "Done."

clean-llvm-bindings:
	rm ${ROOT_DIR}/llvm/llvm.go \
	   ${ROOT_DIR}/llvm/llvm_wrap.c

generate-clang-bindings: ./clang/clang.go
./clang/clang.go: ./clang/clang.i
	@echo "Generating Clang bindings using SWIG..."
	@swig -go -intgosize 64 -cgo \
	-IC:/ProgramData/chocolatey/lib/mingw/tools/install/mingw64/x86_64-w64-mingw32/include \
	-IC:/ProgramData/chocolatey/lib/mingw/tools/install/mingw64/lib/gcc/x86_64-w64-mingw32/12.2.0/include \
	-I${ROOT_DIR}/build/llvm-build/include \
	-I${ROOT_DIR}/thirdparty/llvm-project/clang/include \
	${ROOT_DIR}/clang/clang.i
	@echo "Done."

clean-clang-bindings:
	rm ${ROOT_DIR}/clang/clang.go \
	   ${ROOT_DIR}/clang/clang_wrap.c

build-picolibc: PATH := $(LLVM_BUILD_DIR)/bin:$(PATH)
build-picolibc:
	mkdir -p ./build/picolibc/thumbv7m
	PATH="$(PATH)" CFLAGS="-g" CXXFLAGS="-g" \
	cd ./build/picolibc/thumbv7m && \
	$(ROOT_DIR)/thirdparty/picolibc/scripts/do-clang-thumbv7m-configure \
	-Dtests=false \
	-Dprefix=$(ROOT_DIR)/lib \
	--buildtype=debug && \
	ninja install

build-compiler-rt:
	$(call build-compiler-rt,armv7-none-eabi,armv7,-mthumb)
	$(call build-compiler-rt,armv7m-none-eabi,armv7m,-mthumb)
	$(call build-compiler-rt,armv7em-none-eabi,armv7em,-mthumb)
	$(call build-compiler-rt,armv7s-none-eabi,armv7s,-mthumb)
	$(call build-compiler-rt,armv7k-none-eabi,armv7k,-mthumb)
	$(call build-compiler-rt,armv6m-none-eabi,armv6m,-mthumb)