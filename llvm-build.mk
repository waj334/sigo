# ============================================================================
# llvm-build.mk — CMake-driven LLVM source build for sigo
#
# LLVM is treated as a stable dependency: always built Release, installed to a
# fixed location independent of sigo's own build mode. Toggling sigo between
# Debug and Release does not rebuild LLVM.
#
# Required inputs (set by the including Makefile):
#   ROOT_DIR              — repo root
#   LLVM_SRC_DIR          — path to llvm-project checkout
#   CMAKE_LLVM_COMPONENTS — semicolon-separated list of LLVM_TARGETS_TO_BUILD
#
# Optional inputs:
#   CC, CXX                       — compiler overrides
#   CLANG_TARGET                  — for cross-builds
#   LLVM_CMAKE_BUILD_TYPE         — defaults to Release; override to Debug
#                                   only if you're hacking on LLVM itself
#   LLVM_ENABLE_ASSERTIONS        — defaults to ON; cheap, catches real bugs
#   LLVM_ENABLE_EXPENSIVE_CHECKS  — defaults to OFF; expensive STL debug mode,
#                                   has triggered clang frontend crashes
#   CMAKE_CXX_FLAGS               — extra C++ flags
#   CMAKE_CXX_STANDARD_LIBRARIES  — extra link libs for the LLVM build
#   LD                            — linker selection (ld, lld, gold, mold)
#   NUM_JOBS                      — parallelism (passed to cmake --build via -j)
#
# Outputs:
#   LLVM_INSTALL_DIR  — install prefix for the built LLVM
#                       ($(ROOT_DIR)/build/llvm/install)
#   LLVM_BUILD_DIR    — CMake build tree
#                       ($(ROOT_DIR)/build/llvm/llvm-build)
#
# Targets exposed:
#   configure-llvm     — run cmake to populate the build tree
#   build-llvm         — compile LLVM
#   install-llvm       — install to $(LLVM_INSTALL_DIR)
#   reconfigure-llvm   — drop the CMake cache and re-run configure
#   clean-llvm         — remove the LLVM build tree entirely
# ============================================================================

ifndef LLVM_BUILD_MK_INCLUDED
LLVM_BUILD_MK_INCLUDED := 1

# LLVM is its own world: always Release, fixed location.
LLVM_CMAKE_BUILD_TYPE        ?= Release
LLVM_ENABLE_ASSERTIONS       ?= ON
LLVM_ENABLE_EXPENSIVE_CHECKS ?= OFF

LLVM_INSTALL_DIR := $(ROOT_DIR)/build/llvm/install
LLVM_BUILD_DIR   := $(ROOT_DIR)/build/llvm/llvm-build
LLVM_CMAKE_CACHE := $(LLVM_BUILD_DIR)/CMakeCache.txt

# Compiler args
_LLVM_CMAKE_COMPILER_ARGS :=
ifneq ($(CC),)
	_LLVM_CMAKE_COMPILER_ARGS += -DCMAKE_C_COMPILER=$(CC)
endif
ifneq ($(CXX),)
	_LLVM_CMAKE_COMPILER_ARGS += -DCMAKE_CXX_COMPILER=$(CXX)
endif
ifneq ($(CLANG_TARGET),)
	_LLVM_CMAKE_COMPILER_ARGS += -DCMAKE_C_COMPILER_TARGET=$(CLANG_TARGET) \
	                             -DCMAKE_CXX_COMPILER_TARGET=$(CLANG_TARGET)
endif

# Linker selection — translate $(LD) into CMake's -DCMAKE_LINKER_TYPE
_LLVM_CMAKE_LINKER_ARGS := -DCMAKE_LINKER_TYPE=DEFAULT
ifneq (,$(findstring gold,$(LD)))
	_LLVM_CMAKE_LINKER_ARGS := -DCMAKE_LINKER_TYPE=GOLD
else ifneq (,$(findstring lld,$(LD)))
	_LLVM_CMAKE_LINKER_ARGS := -DCMAKE_LINKER_TYPE=LLD
else ifneq (,$(findstring mold,$(LD)))
	_LLVM_CMAKE_LINKER_ARGS := -DCMAKE_LINKER_TYPE=MOLD
endif

$(LLVM_CMAKE_CACHE):
	@mkdir -p $(LLVM_BUILD_DIR)
	CC=$(CC) CXX=$(CXX) LD=$(LD) cmake -G "Ninja" -B $(LLVM_BUILD_DIR) $(LLVM_SRC_DIR)/llvm \
		-DCMAKE_BUILD_TYPE=$(LLVM_CMAKE_BUILD_TYPE) \
		$(_LLVM_CMAKE_COMPILER_ARGS) \
		$(_LLVM_CMAKE_LINKER_ARGS) \
		-DCMAKE_INSTALL_PREFIX=$(LLVM_INSTALL_DIR) \
		-DCMAKE_CXX_FLAGS="$(CMAKE_CXX_FLAGS)" \
		-DCMAKE_CXX_STANDARD_LIBRARIES="$(CMAKE_CXX_STANDARD_LIBRARIES)" \
		-DLLVM_ENABLE_PROJECTS="llvm;mlir" \
		-DLLVM_ENABLE_ASSERTIONS=$(LLVM_ENABLE_ASSERTIONS) \
		-DLLVM_ENABLE_EXPENSIVE_CHECKS=$(LLVM_ENABLE_EXPENSIVE_CHECKS) \
		-DLLVM_TARGETS_TO_BUILD="$(CMAKE_LLVM_COMPONENTS)" \
		-DLLVM_INSTALL_UTILS=ON \
		-DLLVM_INCLUDE_TESTS=OFF \
		-DMLIR_INCLUDE_TESTS=OFF

.PHONY: configure-llvm build-llvm install-llvm reconfigure-llvm clean-llvm

configure-llvm: $(LLVM_CMAKE_CACHE)

build-llvm: configure-llvm
	cmake --build $(LLVM_BUILD_DIR) $(if $(NUM_JOBS),-j$(NUM_JOBS))

install-llvm: build-llvm
	cmake --install $(LLVM_BUILD_DIR) --prefix $(LLVM_INSTALL_DIR)

reconfigure-llvm:
	rm -f $(LLVM_CMAKE_CACHE)
	$(MAKE) configure-llvm

clean-llvm:
	rm -rf $(LLVM_BUILD_DIR)

endif # LLVM_BUILD_MK_INCLUDED