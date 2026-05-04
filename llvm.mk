# ============================================================================
# llvm.mk — LLVM discovery and flag generation
#
# Inputs (override on command line or before include):
#   LLVM_CONFIG       — explicit path to llvm-config (highest precedence)
#   LLVM_PREFIX       — install prefix (uses $(LLVM_PREFIX)/bin/llvm-config)
#   LLVM_BUILD_DIR    — path to a CMake build tree containing bin/llvm-config
#   LLVM_SRC_DIR      — path to llvm-project source checkout (optional, for
#                       in-tree headers)
#   LLVM_COMPONENTS   — components to link (default: all ARM AVR RISCV passes)
#   LLVM_CXX_STDLIB   — C++ runtime to link (default: -lstdc++)
#   EXECUTABLE_POSTFIX — set by caller for Windows (.exe)
#
# Outputs:
#   LLVM_CONFIG_EXECUTABLE — resolved llvm-config path
#   LLVM_PREFIX            — resolved install prefix
#   LLVM_INCLUDEDIR        — resolved header dir
#   LLVM_LIBDIR            — resolved library dir
#   LLVM_CFLAGS            — cflags to compile against LLVM
#   LLVM_LDFLAGS           — ldflags to link against LLVM
#
# Resolution order (first match wins):
#   1. LLVM_CONFIG    — explicit override
#   2. LLVM_PREFIX    — install prefix
#   3. LLVM_BUILD_DIR — CMake build tree
#
# llvm-config is required. If none of the above resolves to an existing
# executable, this file errors out — LLVM must be built/installed before
# anything that consumes these flags will work.
# ============================================================================

ifndef LLVM_MK_INCLUDED
LLVM_MK_INCLUDED := 1

LLVM_COMPONENTS  ?= all ARM AVR RISCV passes
LLVM_CXX_STDLIB  ?= -lstdc++

ifneq ($(LLVM_CONFIG),)
    LLVM_CONFIG_EXECUTABLE := $(LLVM_CONFIG)
else ifneq ($(LLVM_PREFIX),)
    LLVM_CONFIG_EXECUTABLE := $(LLVM_PREFIX)/bin/llvm-config$(EXECUTABLE_POSTFIX)
else ifneq ($(LLVM_BUILD_DIR),)
    LLVM_CONFIG_EXECUTABLE := $(LLVM_BUILD_DIR)/bin/llvm-config$(EXECUTABLE_POSTFIX)
else
$(error LLVM not found. Set LLVM_PREFIX, LLVM_CONFIG, or LLVM_BUILD_DIR. Build LLVM first if needed (e.g. `make install-llvm`))
endif

ifeq ($(wildcard $(LLVM_CONFIG_EXECUTABLE)),)
$(error llvm-config not found at '$(LLVM_CONFIG_EXECUTABLE)'. Build/install LLVM first or correct LLVM_PREFIX/LLVM_CONFIG)
endif

_LLVM_VERSION := $(shell $(LLVM_CONFIG_EXECUTABLE) --version 2>/dev/null)
ifeq ($(_LLVM_VERSION),)
$(error llvm-config at '$(LLVM_CONFIG_EXECUTABLE)' is not runnable)
endif

# Derive paths from llvm-config
LLVM_PREFIX     ?= $(shell $(LLVM_CONFIG_EXECUTABLE) --prefix)
LLVM_INCLUDEDIR := $(shell $(LLVM_CONFIG_EXECUTABLE) --includedir)
LLVM_LIBDIR     := $(shell $(LLVM_CONFIG_EXECUTABLE) --libdir)

# Optional source-tree headers (in-tree builds only)
ifneq ($(LLVM_SRC_DIR),)
ifneq ($(wildcard $(LLVM_SRC_DIR)/llvm/include),)
	_LLVM_SRC_INCLUDE := -I$(LLVM_SRC_DIR)/llvm/include
endif
endif

LLVM_CFLAGS := \
	$(shell $(LLVM_CONFIG_EXECUTABLE) --cflags) \
	$(_LLVM_SRC_INCLUDE) \
	-I$(LLVM_INCLUDEDIR)

# C++ runtime goes after --end-group; it's the bottom of the dep chain.
LLVM_LDFLAGS := \
	$(shell $(LLVM_CONFIG_EXECUTABLE) --ldflags) \
	-Wl,--start-group \
	$(shell $(LLVM_CONFIG_EXECUTABLE) --libs $(LLVM_COMPONENTS)) \
	-Wl,--end-group \
	$(LLVM_CXX_STDLIB)

.PHONY: print-llvm
print-llvm:
	@echo "LLVM_CONFIG_EXECUTABLE = $(LLVM_CONFIG_EXECUTABLE)"
	@echo "LLVM_VERSION           = $(_LLVM_VERSION)"
	@echo "LLVM_PREFIX            = $(LLVM_PREFIX)"
	@echo "LLVM_INCLUDEDIR        = $(LLVM_INCLUDEDIR)"
	@echo "LLVM_LIBDIR            = $(LLVM_LIBDIR)"
	@echo "LLVM_COMPONENTS        = $(LLVM_COMPONENTS)"
	@echo "LLVM_CXX_STDLIB        = $(LLVM_CXX_STDLIB)"
	@echo "LLVM_CFLAGS			  = $(LLVM_CFLAGS)"
	@echo "LLVM_LDFLAGS			  = $(LLVM_LDFLAGS)"

endif # LLVM_MK_INCLUDED