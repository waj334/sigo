# sysroot.mk -- Build picolibc and compiler-rt builtins per target
#
# Uses clang/LLVM for all cross-compilation.
# picolibc  -> meson
# compiler-rt builtins -> CMake (compiler-rt's own build system)
#
# Usage:
#   $(eval $(call build_sysroot,<name>,<clang-target>,<cflags>,<meson-cpu_family>,<meson-cpu>,<meson-endian>,<cmake-arch>))
#
# Then:
#   make sysroots                    -- build all registered targets
#   make sysroot-armv7em-fp          -- build a single target
#   make clean-sysroots              -- remove all build artifacts

# ---------------------------------------------------------------------
# Toolchain -- override from command line if needed
#
# CMake requires absolute paths for CMAKE_AR/CMAKE_NM/CMAKE_RANLIB.
# Bare names like "llvm-ar" get resolved relative to the source dir
# instead of PATH. We resolve them here.
# ---------------------------------------------------------------------
CLANG       ?= $(shell which clang)
CLANGXX     ?= $(shell which clang++)
LLVM_AR     ?= $(shell which llvm-ar)
LLVM_NM     ?= $(shell which llvm-nm)
LLVM_RANLIB ?= $(shell which llvm-ranlib)
LLVM_STRIP  ?= $(shell which llvm-strip)
CMAKE       ?= cmake

# ---------------------------------------------------------------------
# Paths -- override these from the command line or parent Makefile
# ---------------------------------------------------------------------
PICOLIBC_SRC   ?= $(CURDIR)/thirdparty/picolibc
COMPILERRT_SRC ?= $(CURDIR)/thirdparty/llvm-project/compiler-rt
SYSROOT_OUT    ?= $(CURDIR)/sysroots
BUILD_DIR      ?= $(CURDIR)/build

# Path to LLVM's CMake modules -- needed for standalone compiler-rt builds.
# Default points at the project's own LLVM build directory (where build-llvm
# puts it). Override if using a system LLVM installation.
LLVM_CMAKE_DIR ?= $(CURDIR)/build/Release/llvm-build

# ---------------------------------------------------------------------
# Common flags
# ---------------------------------------------------------------------
COMMON_CFLAGS = \
	-Os \
	-g \
	-ffunction-sections \
	-fdata-sections \
	-fno-exceptions \
	-fno-unwind-tables \
	-fno-asynchronous-unwind-tables \
	-ffreestanding \
	-nostdlib \
	-nostdlibinc

# ---------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------
ALL_SYSROOT_TARGETS :=

# ---------------------------------------------------------------------
# Meson cross-file generator (picolibc)
# ---------------------------------------------------------------------
define generate_meson_cross
	@mkdir -p $(dir $(1))
	@printf "[binaries]\n\
c = '$(CLANG)'\n\
c_ld = 'lld'\n\
ar = '$(LLVM_AR)'\n\
nm = '$(LLVM_NM)'\n\
ranlib = '$(LLVM_RANLIB)'\n\
strip = '$(LLVM_STRIP)'\n\
\n\
[built-in options]\n\
c_args = ['--target=$(2)', $(foreach f,$(3),'$(f)',) '-Os', '-ffunction-sections', '-fdata-sections', '-ffreestanding']\n\
c_link_args = ['--target=$(2)', '-nostdlib', '-fuse-ld=lld']\n\
\n\
[host_machine]\n\
system = 'none'\n\
cpu_family = '$(4)'\n\
cpu = '$(5)'\n\
endian = '$(6)'\n\
" > $(1)
endef

# ---------------------------------------------------------------------
# Compiler-rt toolchain file generator
#
# compiler-rt's builtin-config-ix.cmake uses try_compile to detect
# supported architectures. These try_compile checks only work if
# cross-compilation settings are in a proper toolchain file -- passing
# them as -D flags on the cmake command line does not reliably
# propagate to try_compile.
# ---------------------------------------------------------------------
define generate_crt_toolchain
	@mkdir -p $(dir $(1))
	@printf "\
set(CMAKE_SYSTEM_NAME Generic)\n\
set(CMAKE_SYSTEM_PROCESSOR $(3))\n\
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)\n\
set(CMAKE_C_COMPILER_WORKS 1)\n\
set(CMAKE_CXX_COMPILER_WORKS 1)\n\
set(CMAKE_ASM_COMPILER_WORKS 1)\n\
set(CMAKE_C_COMPILER_TARGET $(2))\n\
set(CMAKE_CXX_COMPILER_TARGET $(2))\n\
set(CMAKE_ASM_COMPILER_TARGET $(2))\n\
" > $(1)
endef

# ---------------------------------------------------------------------
# build_sysroot macro
#
# Arguments:
#   $(1) = target name           (e.g. thumbv7em-hard)
#   $(2) = clang --target triple (e.g. armv7em-none-eabi)
#   $(3) = arch-specific CFLAGS  (e.g. -mthumb -mcpu=cortex-m4 ...)
#   $(4) = meson cpu_family      (e.g. arm, riscv32)
#   $(5) = meson cpu             (e.g. cortex-m4, rv32imac)
#   $(6) = meson endian          (e.g. little, big)
#   $(7) = cmake system proc     (e.g. arm, riscv)
# ---------------------------------------------------------------------
define build_sysroot

# -- Per-target directories ------------------------------------------
# $(strip) is applied inline to every argument. These per-target
# variable names are unique (prefixed by the target name), so they
# don't clobber each other across $(eval) invocations.
$(strip $(1))_BUILD       := $$(BUILD_DIR)/$(strip $(1))
$(strip $(1))_SYSROOT     := $$(SYSROOT_OUT)/$(strip $(1))
$(strip $(1))_PICOLIBC_BD := $$($(strip $(1))_BUILD)/picolibc
$(strip $(1))_CRT_BD      := $$($(strip $(1))_BUILD)/compiler-rt
$(strip $(1))_CROSS_FILE  := $$($(strip $(1))_BUILD)/cross-$(strip $(1)).txt
$(strip $(1))_CRT_TC      := $$($(strip $(1))_BUILD)/crt-toolchain-$(strip $(1)).cmake
$(strip $(1))_TARGET      := $(strip $(2))
$(strip $(1))_ARCHFLAGS   := $(strip $(3))
$(strip $(1))_CPUFAMILY   := $(strip $(4))
$(strip $(1))_CPU         := $(strip $(5))
$(strip $(1))_ENDIAN      := $(strip $(6))
$(strip $(1))_CMAKEARCH   := $(strip $(7))

# Per-target CFLAGS for the Boehm single-file build
$(strip $(1))_CFLAGS := --target=$(strip $(2)) $(strip $(3)) $$(COMMON_CFLAGS)

# -- Meson cross file ------------------------------------------------
$$($(strip $(1))_CROSS_FILE):
	$$(call generate_meson_cross,$$@,$$($(strip $(1))_TARGET),$$($(strip $(1))_ARCHFLAGS),$$($(strip $(1))_CPUFAMILY),$$($(strip $(1))_CPU),$$($(strip $(1))_ENDIAN))

# -- compiler-rt toolchain file -------------------------------------
$$($(strip $(1))_CRT_TC):
	$$(call generate_crt_toolchain,$$@,$(strip $(2)),$(strip $(7)))

# -- picolibc --------------------------------------------------------
$$($(strip $(1))_SYSROOT)/lib/libc.a: $$($(strip $(1))_CROSS_FILE)
	@echo "---- picolibc [$(strip $(1))] ----"
	meson setup $$($(strip $(1))_PICOLIBC_BD) $$(PICOLIBC_SRC) \
		--cross-file $$($(strip $(1))_CROSS_FILE) \
		--prefix=/ \
		-Dmultilib=false \
		-Dspecsdir=none \
		-Dtests=false \
		-Dthread-local-storage=false \
		-Dposix-console=true \
		-Dnewlib-global-atexit=false \
		-Dincludedir=include \
		-Dlibdir=lib
	ninja -C $$($(strip $(1))_PICOLIBC_BD)
	DESTDIR=$$($(strip $(1))_SYSROOT) ninja -C $$($(strip $(1))_PICOLIBC_BD) install

# -- compiler-rt builtins --------------------------------------------
#
# Uses the user's proven standalone compiler-rt build approach:
# - Pass CC/CXX as env vars
# - Set target/flags explicitly via CMAKE_C_COMPILER_TARGET + CMAKE_C_FLAGS
# - Point LLVM_CMAKE_DIR at the installed LLVM CMake modules
# - Use cmake --build --target install
#
$$($(strip $(1))_SYSROOT)/lib/libclang_rt.builtins.a: $$($(strip $(1))_CRT_TC) $$($(strip $(1))_SYSROOT)/lib/libc.a
	@echo "---- compiler-rt [$(strip $(1))] ----"
	CC=$$(CLANG) CXX=$$(CLANGXX) $$(CMAKE) $$(COMPILERRT_SRC) \
		-G Ninja \
		-B $$($(strip $(1))_CRT_BD) \
		-DCMAKE_TOOLCHAIN_FILE=$$($(strip $(1))_CRT_TC) \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_INSTALL_PREFIX=$$($(strip $(1))_SYSROOT) \
		-DCMAKE_SYSROOT=$$($(strip $(1))_SYSROOT) \
		-DBUILD_SHARED_LIBS=OFF \
		-DCMAKE_AR=$$(LLVM_AR) \
		-DCMAKE_NM=$$(LLVM_NM) \
		-DCMAKE_RANLIB=$$(LLVM_RANLIB) \
		-DCMAKE_C_COMPILER=$$(CLANG) \
		-DCMAKE_C_FLAGS="-nostdlib $(strip $(3))" \
		-DCMAKE_CXX_COMPILER=$$(CLANGXX) \
		-DCMAKE_CXX_FLAGS="-nostdlib $(strip $(3))" \
		-DCMAKE_ASM_FLAGS="$(strip $(3))" \
		-DLLVM_CMAKE_DIR=$$(LLVM_CMAKE_DIR) \
		-DCOMPILER_RT_OS_DIR="$(strip $(2))" \
		-DCOMPILER_RT_DEFAULT_TARGET_ONLY=ON \
		-DCOMPILER_RT_BAREMETAL_BUILD=ON \
		-DCOMPILER_RT_BUILD_BUILTINS=ON \
		-DCOMPILER_RT_BUILD_CRT=ON \
		-DCOMPILER_RT_BUILD_SANITIZERS=OFF \
		-DCOMPILER_RT_BUILD_XRAY=OFF \
		-DCOMPILER_RT_BUILD_LIBFUZZER=OFF \
		-DCOMPILER_RT_BUILD_PROFILE=OFF \
		-DCOMPILER_RT_BUILD_MEMPROF=OFF \
		-DCOMPILER_RT_BUILD_ORC=OFF \
		-DCOMPILER_RT_BUILD_CTX_PROFILE=OFF \
		-DCOMPILER_RT_BUILD_GWP_ASAN=OFF \
		-DCOMPILER_RT_INCLUDE_TESTS=OFF
	$$(CMAKE) --build $$($(strip $(1))_CRT_BD) --target install
	@# The installed name includes an arch suffix -- create a
	@# predictable name so the linker can use -lclang_rt.builtins
	@if [ ! -f "$$@" ]; then \
		installed=$$$$(find $$($(strip $(1))_SYSROOT) -name 'libclang_rt.builtins*.a' -print -quit); \
		if [ -n "$$$$installed" ]; then \
			ln -sf "$$$$installed" "$$@"; \
		else \
			echo "ERROR: compiler-rt built no library for $(strip $(1))"; \
			exit 1; \
		fi; \
	fi

# -- Per-target phony ------------------------------------------------
.PHONY: sysroot-$(strip $(1)) clean-sysroot-$(strip $(1))

sysroot-$(strip $(1)): \
	$$($(strip $(1))_SYSROOT)/lib/libc.a \
	$$($(strip $(1))_SYSROOT)/lib/libclang_rt.builtins.a
	@echo "---- sysroot ready: $$($(strip $(1))_SYSROOT) ----"

clean-sysroot-$(strip $(1)):
	rm -rf $$($(strip $(1))_BUILD) $$($(strip $(1))_SYSROOT)

ALL_SYSROOT_TARGETS += sysroot-$(strip $(1))

endef  # build_sysroot


# ---------------------------------------------------------------------
# Register targets
#
# Sysroots are keyed on (triple + FP variant), NOT on CPU name.
# Multiple CPUs share the same sysroot when they have identical
# triple and FPU configurations:
#
#   armv6m-nofp        -> Cortex-M0, M0+, M1
#   armv7m-nofp        -> Cortex-M3
#   armv7em-fp         -> Cortex-M4, M7
#   armv7em-nofp       -> Cortex-M4/M7 software float
#   armv8m.base-nofp   -> Cortex-M23
#   armv8m.main-fp     -> Cortex-M33, M35P
#   armv81m.main-fp    -> Cortex-M52, M55, M85
#
# The cflags use -march= with +fp/+nofp only -- no explicit -mfpu.
# Clang infers the correct FPU from the march string. Adding -mfpu
# can conflict with picolibc's assembly (setjmp.S uses d-register
# save/restore that clang rejects under SP-only FPU constraints).
# ---------------------------------------------------------------------
#                      name                    clang-target                 cflags                                cpu_family  cpu          endian  cmake-arch
$(eval $(call build_sysroot,armv6m-nofp,       armv6m-none-eabi,            -mthumb -march=armv6m+nofp,           arm,        cortex-m0+,  little, arm))
$(eval $(call build_sysroot,armv7m-nofp,       armv7m-none-eabi,            -mthumb -march=armv7m+nofp,           arm,        cortex-m3,   little, arm))
$(eval $(call build_sysroot,armv7em-nofp,      armv7em-none-eabi,           -mthumb -march=armv7em+nofp,          arm,        cortex-m4,   little, arm))
$(eval $(call build_sysroot,armv7em-fp,        armv7em-none-eabi,         	-mthumb -march=armv7em+fp,            arm,        cortex-m4,   little, arm))
$(eval $(call build_sysroot,armv8m.base-nofp,  armv8m.base-none-eabi,       -mthumb -march=armv8m.base+nofp,      arm,        cortex-m23,  little, arm))
$(eval $(call build_sysroot,armv8m.main-fp,    armv8m.main-none-eabi,     	-mthumb -march=armv8m.main+fp,        arm,        cortex-m33,  little, arm))
#$(eval $(call build_sysroot,armv81m.main-fp,   armv81m.main-none-eabi,    -mthumb -march=armv8.1m.main+fp,      arm,        cortex-m55,  little, arm))
#$(eval $(call build_sysroot,riscv32-imac,     riscv32-none-elf,             -march=rv32imac -mabi=ilp32,          riscv32,    rv32imac,    little, riscv32))

# ---------------------------------------------------------------------
# Aggregate targets
# ---------------------------------------------------------------------
.PHONY: sysroots clean-sysroots

sysroots: $(ALL_SYSROOT_TARGETS)

clean-sysroots:
	rm -rf $(BUILD_DIR) $(SYSROOT_OUT)