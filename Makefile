ROOT_DIR := $(patsubst %/,%,$(dir $(realpath $(lastword $(MAKEFILE_LIST)))))

# ----------------------------------------------------------------------------
# Build mode (applies to sigo and GoIR; LLVM is always Release)
# ----------------------------------------------------------------------------
SIGO_BUILD_RELEASE ?= 0
ifeq ($(SIGO_BUILD_RELEASE),0)
    CMAKE_BUILD_TYPE := Debug
    CGO_CFLAGS  += -O0 -g
    CGO_LDFLAGS += -g
else
    CMAKE_BUILD_TYPE := Release
    CGO_CFLAGS  += -Oz
endif

# ----------------------------------------------------------------------------
# Linker selection
#
# Default is plain `ld`. Override for faster links:
#   make LD=lld build-llvm        # ~10-20x faster than GNU ld for LLVM
#   make LD=mold build-llvm       # often another 2-3x faster than lld
#
# Applies to LLVM and GoIR CMake builds and to the final sigoc link.
# ----------------------------------------------------------------------------
LD ?= ld

ifneq ($(LD),ld)
	APP_LDFLAGS += -fuse-ld=$(patsubst ld.%,%,$(LD))
endif

# ----------------------------------------------------------------------------
# Platform
# ----------------------------------------------------------------------------
ifeq ($(OS),Windows_NT)
    EXECUTABLE_POSTFIX := .exe
    APP_LDFLAGS += -lole32 -luuid -lpsapi -lshell32 -ladvapi32 -lntdll
    CMAKE_CXX_FLAGS += -pthread -femulated-tls
    CMAKE_CXX_STANDARD_LIBRARIES += -lpthread
    CLANG_TARGET := x86_64-pc-windows-gnu
    CC  ?= clang
    CXX ?= clang++
    # NOTE: ld should be replaced with ld.lld directly on Windows since Go is dumb.
else
    APP_LDFLAGS += -lrt -ldl -lpthread -lm -lz -ltinfo -lzstd
endif

# ----------------------------------------------------------------------------
# Paths
#
# LLVM lives at a fixed, mode-independent location (always Release).
# GoIR + sigo install vary by build mode.
# ----------------------------------------------------------------------------
LLVM_SRC_DIR     := $(ROOT_DIR)/thirdparty/llvm-project
GOIR_INSTALL_DIR := $(ROOT_DIR)/build/$(CMAKE_BUILD_TYPE)/install
INSTALL_DIR      := $(GOIR_INSTALL_DIR)  # alias for sigo-side reference
BINDIR           := ./bin
ABS_BINDIR       := $(ROOT_DIR)/bin

# Tell llvm.mk where to find LLVM. This must come before the include.
LLVM_PREFIX     ?= $(ROOT_DIR)/build/llvm/install
LLVM_CONFIG		?= $(LLVM_PREFIX)/bin/llvm-config
LLVM_COMPONENTS ?= all ARM AVR RISCV passes

# ----------------------------------------------------------------------------
# Targets to build into LLVM (LLVM_TARGETS_TO_BUILD)
# ----------------------------------------------------------------------------
LLVM_BUILD_COMPONENTS := ARM AVR RISCV

empty :=
space := $(empty) $(empty)
CMAKE_LLVM_COMPONENTS := $(subst $(space),;,$(strip $(LLVM_BUILD_COMPONENTS)))

# ----------------------------------------------------------------------------
# Pull in toolchain plumbing
#
# *-build.mk files are always included (they only define targets).
# llvm.mk + mlir.mk derive flags from llvm-config and are only included once
# LLVM is actually installed — otherwise their flag derivation would error
# out before LLVM gets a chance to be built.
# ----------------------------------------------------------------------------
include $(ROOT_DIR)/llvm-build.mk
include $(ROOT_DIR)/goir-build.mk

ifneq ($(wildcard $(LLVM_CONFIG)$(EXECUTABLE_POSTFIX)),)
    include $(ROOT_DIR)/llvm.mk
    include $(ROOT_DIR)/mlir.mk
    HAVE_LLVM := 1
endif

# Convenience: configure or reconfigure the whole stack
.PHONY: configure reconfigure
configure: configure-llvm configure-goir
reconfigure: reconfigure-llvm reconfigure-goir

# ----------------------------------------------------------------------------
# Compose final CGo flags
# ----------------------------------------------------------------------------
ifdef HAVE_LLVM
    CGO_CFLAGS += -fPIC -ffunction-sections -fdata-sections \
                  $(LLVM_CFLAGS) $(MLIR_CFLAGS)

    # GoIR + MLIR + LLVM, in dep order. GoIR depends on MLIR which depends
    # on LLVM, so listing them in that order keeps each --start/--end-group
    # resolution localized.
    CGO_LDFLAGS += -Wl,--gc-sections \
                   $(MLIR_LDFLAGS) \
                   $(LLVM_LDFLAGS) \
                   $(APP_LDFLAGS)

    # A few LLVM libs that --libs doesn't always pull in but sigo needs
    CGO_LDFLAGS += -lLLVMTableGen -lLLVMOption -lLLVMPlugins

    CGO_CXXFLAGS := -std=c++17 -fno-rtti $(CGO_CFLAGS)
endif

# ----------------------------------------------------------------------------
# Sources
# ----------------------------------------------------------------------------
GO_BUILDER_SRCS      		:= $(wildcard $(ROOT_DIR)/builder/*.go)
GO_COMPILER_SSA_SRCS 		:= $(wildcard $(ROOT_DIR)/compiler/ssa/*.go)
GO_CMD_SIGOC_SRCS    		:= $(wildcard $(ROOT_DIR)/cmd/sigoc/*.go)
GO_SRCS 					:= $(GO_BUILDER_SRCS) $(GO_COMPILER_SSA_SRCS) $(GO_CMD_SIGOC_SRCS)

GO_CMD_TABLEGEN_CSP_SRCS    := $(wildcard $(ROOT_DIR)/cmd/tablegen-csp/*.go)

TARGETS_DEVICE_SRCS += $(wildcard $(ROOT_DIR)/targets/device/*.go)
TARGETS_DEVICE_SRCS += $(wildcard $(ROOT_DIR)/targets/device/importer/*.go)
TARGETS_DEVICE_SRCS += $(wildcard $(ROOT_DIR)/targets/device/importer/atdf/*.go)
TARGETS_DEVICE_SRCS += $(wildcard $(ROOT_DIR)/targets/device/importer/svd/*.go)
TARGETS_DEVICE_SRCS += $(wildcard $(ROOT_DIR)/targets/device/svd/*.go)

_GOIR_LIBS    := $(wildcard $(GOIR_INSTALL_DIR)/lib/*.a)
_GOIR_LDFLAGS := -L$(GOIR_INSTALL_DIR)/lib \
				 -Wl,--start-group \
                 $(sort $(patsubst lib%.a,-l%,$(notdir $(_GOIR_LIBS)))) \
                 -Wl,--end-group

CGO_CFLAGS    += -I$(GOIR_INSTALL_DIR)/include
CGO_CXXFLAGS  += -I$(GOIR_INSTALL_DIR)/include
CGO_LDFLAGS   += $(_GOIR_LDFLAGS)

# ----------------------------------------------------------------------------
# Executables
# ----------------------------------------------------------------------------
SIGO_EXE     := $(BINDIR)/sigoc$(EXECUTABLE_POSTFIX)
ABS_SIGO_EXE := $(ABS_BINDIR)/sigoc$(EXECUTABLE_POSTFIX)

CSP_GEN_EXE      := $(BINDIR)/csp-gen$(EXECUTABLE_POSTFIX)
DEF_GEN_EXE      := $(BINDIR)/def-gen$(EXECUTABLE_POSTFIX)
TBDEF_GEN_EXE    := $(BINDIR)/tbdef-gen$(EXECUTABLE_POSTFIX)
TABLEGEN_CSP_EXE := $(BINDIR)/tablegen-csp$(EXECUTABLE_POSTFIX)
BIN2STR_EXE      := $(BINDIR)/bin2str$(EXECUTABLE_POSTFIX)

SSA_TEST_EXE     := $(BINDIR)/ssa_test$(EXECUTABLE_POSTFIX)
ABS_SSA_TEST_EXE := $(ABS_BINDIR)/ssa_test$(EXECUTABLE_POSTFIX)

DEBUG ?= 0

# ----------------------------------------------------------------------------
# Test helpers
# ----------------------------------------------------------------------------
define build-test
	@rm -f $(1)$(EXECUTABLE_POSTFIX)
	CGO_CFLAGS="$(CGO_CFLAGS)" CGO_LDFLAGS="$(CGO_LDFLAGS)" \
		go test -gcflags "all=-N -l" \
		-ldflags="-linkmode external -extldflags=-Wl,--allow-multiple-definition" \
		-c -o $(1) $(2)
endef

define run-test
	PATH="$(GOIR_INSTALL_DIR)/bin:$(LLVM_PREFIX)/bin:$(PATH)" \
	CGO_CFLAGS="$(CGO_CFLAGS)" CGO_LDFLAGS="$(CGO_LDFLAGS)" \
		go test -v -gcflags "all=-N -l" \
		-ldflags="-linkmode external -extldflags=-Wl,--allow-multiple-definition" \
		$(1) -args $(args)
endef

# ----------------------------------------------------------------------------
# Top-level targets
# ----------------------------------------------------------------------------
.PHONY: all sigo debug release env clean clean-sigo \
        ssa_test build-tests run-tests test clean-tests

all: sigo

env:
	@CGO_CFLAGS="$(CGO_CFLAGS)" CGO_LDFLAGS="$(CGO_LDFLAGS)" CGO_CXXFLAGS="$(CGO_CXXFLAGS)" go env

clean: clean-sigo clean-tests clean-sysroots clean-goir clean-llvm

# ----------------------------------------------------------------------------
# sigoc
# ----------------------------------------------------------------------------
$(SIGO_EXE): install-goir sysroots $(GO_SRCS) $(_GOIR_LIBS)
	rm -f $(SIGO_EXE)
ifeq ($(SIGO_BUILD_RELEASE),1)
	CGO_CFLAGS="$(CGO_CFLAGS)" CGO_LDFLAGS="$(CGO_LDFLAGS)" CGO_CXXFLAGS="$(CGO_CXXFLAGS)" \
		go build -o $(SIGO_EXE) -ldflags="-linkmode external" $(ROOT_DIR)/cmd/sigoc
else
	CGO_CFLAGS="$(CGO_CFLAGS)" CGO_LDFLAGS="$(CGO_LDFLAGS)" CGO_CXXFLAGS="$(CGO_CXXFLAGS)" \
		go build -o $(SIGO_EXE) -gcflags "all=-N -l" -ldflags="-linkmode external" $(ROOT_DIR)/cmd/sigoc
endif

sigo: $(SIGO_EXE)

debug: sigo
	dlv --listen=:2346 --headless=true --api-version=2 --accept-multiclient exec $(SIGO_EXE) -- $(args)

release: sigo

clean-sigo:
	rm -f $(SIGO_EXE)

# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
$(SSA_TEST_EXE): $(GO_COMPILER_SSA_SRCS) $(_GOIR_LIBS)
	$(call build-test, $@, ./compiler/ssa)

ssa_test: $(SSA_TEST_EXE)

build-tests: ssa_test

run-tests:
	$(call run-test,./compiler/ssa)

test: build-tests
ifeq ($(DEBUG),1)
	dlv --listen=:2346 --headless=true --api-version=2 --accept-multiclient \
		exec --wd=$(ROOT_DIR)/compiler/ssa $(ABS_SSA_TEST_EXE) -- -tests=$(TESTS)
else
	cd ./compiler/ssa && $(ABS_SSA_TEST_EXE) -tests=$(TESTS)
endif

clean-tests:
	rm -f $(SSA_TEST_EXE)

# ----------------------------------------------------------------------------
# Codegen tools
# ----------------------------------------------------------------------------
.PHONY: tbdef-gen tablegen-csp

$(TBDEF_GEN_EXE): $(wildcard $(ROOT_DIR)/cmd/tbdef-gen/*.go) $(TARGETS_DEVICE_SRCS)
ifeq ($(SIGO_BUILD_RELEASE),1)
	go build -o $(TBDEF_GEN_EXE) $(ROOT_DIR)/cmd/tbdef-gen
else
	go build -o $(TBDEF_GEN_EXE) -gcflags "all=-N -l" $(ROOT_DIR)/cmd/tbdef-gen
endif

tbdef-gen: $(TBDEF_GEN_EXE)
ifeq ($(DEBUG),1)
	dlv --listen=:2346 --headless=true --api-version=2 --accept-multiclient exec $(TBDEF_GEN_EXE) -- $(args)
endif

$(TABLEGEN_CSP_EXE): $(GO_SRCS) $(_GOIR_LIBS) $(GO_CMD_TABLEGEN_CSP_SRCS)
	rm -f $(TABLEGEN_CSP_EXE)
ifeq ($(SIGO_BUILD_RELEASE),1)
	CGO_CFLAGS="$(CGO_CFLAGS)" CGO_LDFLAGS="$(CGO_LDFLAGS)" CGO_CXXFLAGS="$(CGO_CXXFLAGS)" \
		go build -o $(TABLEGEN_CSP_EXE) -ldflags="-linkmode external" $(ROOT_DIR)/cmd/tablegen-csp
else
	CGO_CFLAGS="$(CGO_CFLAGS)" CGO_LDFLAGS="$(CGO_LDFLAGS)" CGO_CXXFLAGS="$(CGO_CXXFLAGS)" \
		go build -o $(TABLEGEN_CSP_EXE) -gcflags "all=-N -l" -ldflags="-linkmode external" $(ROOT_DIR)/cmd/tablegen-csp
endif

tablegen-csp: $(TABLEGEN_CSP_EXE)
ifeq ($(DEBUG),1)
	dlv --listen=:2346 --headless=true --api-version=2 --accept-multiclient exec $(TABLEGEN_CSP_EXE) -- $(args)
endif

# ----------------------------------------------------------------------------
# Sysroots
# ----------------------------------------------------------------------------
include sysroot.mk