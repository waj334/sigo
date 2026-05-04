# ============================================================================
# goir-build.mk — CMake-driven build of the GoIR MLIR dialect for sigo
#
# GoIR is developed alongside sigo, so it follows sigo's build mode:
# Debug builds of sigo get a Debug GoIR, Release gets Release GoIR. The
# install location varies accordingly.
#
# GoIR depends on a fully installed LLVM/MLIR (see llvm-build.mk).
# install-llvm must run before configure-goir.
#
# Required inputs (set by the including Makefile):
#   ROOT_DIR          — repo root
#   GOIR_INSTALL_DIR  — install prefix (typically build/$(CMAKE_BUILD_TYPE)/install)
#   CMAKE_BUILD_TYPE  — Debug / Release / etc. (sigo's mode)
#   LLVM_INSTALL_DIR  — where LLVM was installed (from llvm-build.mk)
#
# Optional inputs:
#   CC, CXX                       — compiler overrides
#   CLANG_TARGET                  — for cross-builds
#   CMAKE_CXX_FLAGS               — extra C++ flags
#   CMAKE_CXX_STANDARD_LIBRARIES  — extra link libs for the GoIR build
#   LD                            — linker selection (ld, lld, gold, mold)
#   NUM_JOBS                      — parallelism (passed to cmake --build via -j)
#
# Outputs:
#   GOIR_BUILD_DIR    — CMake build tree
#                       ($(ROOT_DIR)/build/$(CMAKE_BUILD_TYPE)/goir-build)
#   GOIR_LIB          — $(GOIR_INSTALL_DIR)/lib/libGoIR.a
#
# Targets exposed:
#   configure-goir     — run cmake to populate the build tree
#   build-goir         — compile GoIR
#   install-goir       — install to $(GOIR_INSTALL_DIR)
#   reconfigure-goir   — drop the CMake cache and re-run configure
#   clean-goir         — remove the GoIR build tree entirely
# ============================================================================

ifndef GOIR_BUILD_MK_INCLUDED
GOIR_BUILD_MK_INCLUDED := 1

GOIR_ROOT        := $(ROOT_DIR)/goir
GOIR_BUILD_DIR   := $(ROOT_DIR)/build/$(CMAKE_BUILD_TYPE)/goir-build
GOIR_CMAKE_CACHE := $(GOIR_BUILD_DIR)/CMakeCache.txt
GOIR_LIB         := $(GOIR_INSTALL_DIR)/lib/libGoIR.a

# Compiler args
_GOIR_CMAKE_COMPILER_ARGS :=
ifneq ($(CC),)
	_GOIR_CMAKE_COMPILER_ARGS += -DCMAKE_C_COMPILER=$(CC)
endif
ifneq ($(CXX),)
	_GOIR_CMAKE_COMPILER_ARGS += -DCMAKE_CXX_COMPILER=$(CXX)
endif
ifneq ($(CLANG_TARGET),)
	_GOIR_CMAKE_COMPILER_ARGS += -DCMAKE_C_COMPILER_TARGET=$(CLANG_TARGET) \
	                             -DCMAKE_CXX_COMPILER_TARGET=$(CLANG_TARGET)
endif

# Linker selection — translate $(LD) into CMake's -DCMAKE_LINKER_TYPE
_GOIR_CMAKE_LINKER_ARGS := -DCMAKE_LINKER_TYPE=DEFAULT
ifneq (,$(findstring gold,$(LD)))
	_GOIR_CMAKE_LINKER_ARGS := -DCMAKE_LINKER_TYPE=GOLD
else ifneq (,$(findstring lld,$(LD)))
	_GOIR_CMAKE_LINKER_ARGS := -DCMAKE_LINKER_TYPE=LLD
else ifneq (,$(findstring mold,$(LD)))
	_GOIR_CMAKE_LINKER_ARGS := -DCMAKE_LINKER_TYPE=MOLD
endif

$(GOIR_CMAKE_CACHE):
	@mkdir -p $(GOIR_BUILD_DIR)
	CC=$(CC) CXX=$(CXX) LD=$(LD) cmake -G "Ninja" -B $(GOIR_BUILD_DIR) $(GOIR_ROOT) \
		-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		$(_GOIR_CMAKE_COMPILER_ARGS) \
		$(_GOIR_CMAKE_LINKER_ARGS) \
		-DCMAKE_INSTALL_PREFIX=$(GOIR_INSTALL_DIR) \
		-DCMAKE_CXX_FLAGS="$(CMAKE_CXX_FLAGS)" \
		-DCMAKE_CXX_STANDARD_LIBRARIES="$(CMAKE_CXX_STANDARD_LIBRARIES)" \
		-DCMAKE_PREFIX_PATH="$(LLVM_INSTALL_DIR)/lib/cmake;$(GOIR_INSTALL_DIR)/lib/cmake"

.PHONY: configure-goir build-goir install-goir reconfigure-goir clean-goir

configure-goir: install-llvm $(GOIR_CMAKE_CACHE)

$(GOIR_LIB): configure-goir
	cmake --build $(GOIR_BUILD_DIR) $(if $(NUM_JOBS),-j$(NUM_JOBS))

build-goir: $(GOIR_LIB)

install-goir: build-goir
	cmake --install $(GOIR_BUILD_DIR) --prefix $(GOIR_INSTALL_DIR)

reconfigure-goir:
	rm -f $(GOIR_CMAKE_CACHE)
	$(MAKE) configure-goir

clean-goir:
	rm -rf $(GOIR_BUILD_DIR)

endif # GOIR_BUILD_MK_INCLUDED