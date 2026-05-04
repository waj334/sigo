# ============================================================================
# mlir.mk — MLIR discovery and flag generation
#
# Requires llvm.mk to be included first (uses LLVM_LIBDIR, LLVM_INCLUDEDIR,
# LLVM_BUILD_DIR, LLVM_SRC_DIR).
#
# MLIR ships no mlir-config, so libraries are discovered by globbing the
# resolved LLVM library directory.
#
# Inputs (optional overrides):
#   MLIR_LIBS         — explicit space-separated list of -l flags to link
#                       (skip the glob if set)
#   MLIR_LIB_PATTERN  — glob pattern for MLIR archives (default: libMLIR*.a)
#
# Outputs:
#   MLIR_CFLAGS  — cflags to compile against MLIR
#   MLIR_LDFLAGS — ldflags to link against MLIR (wrapped in --start/--end-group)
# ============================================================================

ifndef MLIR_MK_INCLUDED
MLIR_MK_INCLUDED := 1

ifndef LLVM_MK_INCLUDED
$(error mlir.mk requires llvm.mk to be included first)
endif

MLIR_LIB_PATTERN ?= libMLIR*.a

# In-tree builds expose generated TableGen headers under tools/mlir/include
ifneq ($(LLVM_BUILD_DIR),)
ifneq ($(wildcard $(LLVM_BUILD_DIR)/tools/mlir/include),)
	_MLIR_BUILD_INCLUDE := -I$(LLVM_BUILD_DIR)/tools/mlir/include
endif
endif

# Source-tree headers (in-tree builds)
ifneq ($(LLVM_SRC_DIR),)
ifneq ($(wildcard $(LLVM_SRC_DIR)/mlir/include),)
	_MLIR_SRC_INCLUDE := -I$(LLVM_SRC_DIR)/mlir/include
endif
endif

# Resolve MLIR libraries
ifneq ($(MLIR_LIBS),)
	_MLIR_LIBS := $(MLIR_LIBS)
else
	_MLIR_LIB_FILES := $(wildcard $(LLVM_LIBDIR)/$(MLIR_LIB_PATTERN))
	ifeq ($(_MLIR_LIB_FILES),)
$(warning No MLIR libraries found matching $(LLVM_LIBDIR)/$(MLIR_LIB_PATTERN))
	endif
	_MLIR_LIBS := $(sort $(patsubst lib%.a,-l%,$(notdir $(_MLIR_LIB_FILES))))
endif

MLIR_CFLAGS := \
	$(_MLIR_SRC_INCLUDE) \
	$(_MLIR_BUILD_INCLUDE)

MLIR_LDFLAGS := \
	-L$(LLVM_LIBDIR) \
	-Wl,--start-group \
	$(_MLIR_LIBS) \
	-Wl,--end-group

.PHONY: print-mlir
print-mlir:
	@echo "MLIR_LIB_PATTERN  = $(MLIR_LIB_PATTERN)"
	@echo "MLIR libs found   = $(words $(_MLIR_LIBS))"
	@echo "MLIR_CFLAGS       = $(MLIR_CFLAGS)"

endif # MLIR_MK_INCLUDED