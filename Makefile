ROOT_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
LLVM_BUILD_DIR=${ROOT_DIR}/build/llvm-build
ifeq ($(OS),Windows_NT)
	EXECUTABLE_POSTFIX=.exe
	CGO_LDFLAGS += -lole32 -luuid -lpsapi -lshell32 -ladvapi32
endif
C_FOR_GO_EXECUTABLE=c-for-go${EXECUTABLE_POSTFIX}
LLVM_CONFIG_EXECUTABLE=${ROOT_DIR}/build/llvm-build/bin/llvm-config${EXECUTABLE_POSTFIX}

CGO_LDFLAGS += -O2 -g $(shell ${LLVM_CONFIG_EXECUTABLE} --ldflags) $(shell ${LLVM_CONFIG_EXECUTABLE} --libs)
CGO_CFLAGS += -O2 -g -fPIC $(shell ${LLVM_CONFIG_EXECUTABLE} --cflags)

SIGO_EXECUTABLE=sigoc${EXECUTABLE_POSTFIX}

# These LLVM header will be preprocessed before handing them to SWIG
LLVM_PREPROCESS_HEADERS += llvm-c/Target.h llvm-c/TargetMachine.h
LLVM_PREPROCESS_HEADERS_OUT := ${ROOT_DIR}/build/llvm-headers/include

.PHONY: all clean sigo clean-sigo configure-llvm build-llvm generate-llvm-bindings clean-llvm-bindings

all: sigo

clean: clean-sigo clean-llvm-bindings

sigo: ./llvm/llvm.go
	CGO_CFLAGS="${CGO_CFLAGS}" CGO_LDFLAGS="${CGO_LDFLAGS} -lstdc++" go build -o ${ROOT_DIR}/bin/${SIGO_EXECUTABLE} -gcflags "all=-N -l" ${ROOT_DIR}/cmd/sigoc

debug: sigo
	dlv --listen=:2346 --headless=true --api-version=2 --accept-multiclient exec ${ROOT_DIR}/bin/${SIGO_EXECUTABLE} -- ${args}

build-test:
	@rm -f ./bin/test${EXECUTABLE_POSTFIX}
	CGO_CFLAGS="${CGO_CFLAGS}" CGO_LDFLAGS="${CGO_LDFLAGS} -lstdc++" go test -gcflags "all=-N -l" -c -o ./bin/test${EXECUTABLE_POSTFIX} ${package}

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
		-DLLVM_ENABLE_ASSERTIONS=ON

build-llvm: configure-llvm
	cmake --build ${LLVM_BUILD_DIR}
	make preprocess-llvm-headers

preprocess-llvm-headers:
	$(foreach item,$(LLVM_PREPROCESS_HEADERS),$(call remove-typedefs,$(item)))

define remove-typedefs
	@echo "Preprocessing $(1)"
	@mkdir -p $(dir ${LLVM_PREPROCESS_HEADERS_OUT}/$(1))
	@sed '/typedef struct LLVM/d' ${ROOT_DIR}/thirdparty/llvm-project/llvm/include/$(1) > ${LLVM_PREPROCESS_HEADERS_OUT}/$(1)

endef

generate-llvm-bindings: ./llvm/llvm.go
./llvm/llvm.go: ./llvm/llvm.i ./llvm/argnames.i ./build/llvm-headers/include/llvm-c
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