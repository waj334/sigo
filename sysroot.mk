# sysroot.mk -- Build picolibc, compiler-rt builtins, and lwIP per target
#
# All three components are static-library cross-builds. The compiler must
# be clang (we use --target=, -mthumb, -march=, etc. clang-style flags),
# but it can be any clang the developer wants:
#
#   SYSROOT_CC          path to clang        (defaults to $(CC) if clang,
#                                             else `which clang`)
#   SYSROOT_CXX         path to clang++      (defaults to $(CXX) if clang,
#                                             else `which clang++`)
#   SYSROOT_LD          linker name passed   (defaults to $(LD); ld/lld/
#                       to -fuse-ld=          gold/mold)
#
# Binutils default to whatever ships next to SYSROOT_CC -- if SYSROOT_CC
# is /opt/llvm-20/bin/clang, the default SYSROOT_AR is
# /opt/llvm-20/bin/llvm-ar. Each can be overridden:
#
#   SYSROOT_AR / SYSROOT_NM / SYSROOT_RANLIB / SYSROOT_STRIP
#
# Compiler-rt isolation
# ---------------------
# clang ships with its own libclang_rt.builtins-*.a in its resource dir
# (typically /usr/lib/clang/<ver>/). When clang drives a link, it
# auto-appends those builtins. To make sure downstream consumers (sigo,
# lwIP, etc.) pick up OUR compiler-rt instead of the chosen toolchain's
# bundled one, we override clang's resource dir per-target.
#
# A clang resource directory contains TWO things:
#   1. lib/<triple>/libclang_rt.builtins.a    -- the runtime archive
#   2. include/                               -- compiler-builtin headers
#                                                (float.h, stddef.h, ...)
#
# We populate (1) ourselves from our compiler-rt build. For (2) we
# symlink clang's own include/ tree into our resource dir, so the
# compiler-builtin headers continue to come from clang while the
# builtins library comes from us.
#
# Build order matters: compiler-rt must be installed (and the resource
# dir laid out) before lwIP runs, because lwIP's CMake try_compile()
# could pick up the chosen toolchain's bundled builtins if ours aren't
# in place yet. The dep chain enforces this.

# ---------------------------------------------------------------------
# Toolchain selection
# ---------------------------------------------------------------------

# If $(CC) looks like clang, use it; otherwise find a system clang.
# Same for $(CXX). This keeps the default sane when the main Makefile
# leaves CC unset (implicit "cc" = GCC, which isn't a cross-compiler).
ifneq (,$(findstring clang,$(CC)))
	SYSROOT_CC ?= $(CC)
else
	SYSROOT_CC ?= $(shell which clang)
endif

ifneq (,$(findstring clang,$(CXX)))
	SYSROOT_CXX ?= $(CXX)
else
	SYSROOT_CXX ?= $(shell which clang++)
endif

# Linker name passed via -fuse-ld (ld, lld, gold, mold). Tracks $(LD).
SYSROOT_LD ?= $(LD)

ifeq ($(SYSROOT_CC),)
$(error No clang found. Set SYSROOT_CC explicitly or install clang.)
endif

# Toolchain binutils default to whatever ships next to SYSROOT_CC.
# If SYSROOT_CC is /opt/llvm-20/bin/clang, these resolve to
# /opt/llvm-20/bin/llvm-ar etc.
_SYSROOT_TOOLCHAIN_BIN := $(dir $(realpath $(SYSROOT_CC)))

SYSROOT_AR     ?= $(_SYSROOT_TOOLCHAIN_BIN)llvm-ar
SYSROOT_NM     ?= $(_SYSROOT_TOOLCHAIN_BIN)llvm-nm
SYSROOT_RANLIB ?= $(_SYSROOT_TOOLCHAIN_BIN)llvm-ranlib
SYSROOT_STRIP  ?= $(_SYSROOT_TOOLCHAIN_BIN)llvm-strip

# Clang's default resource dir, used as the source for compiler-builtin
# headers (float.h, stddef.h, ...) which we symlink into our per-target
# resource dirs.
SYSROOT_CLANG_RESOURCE_DIR ?= $(shell $(SYSROOT_CC) -print-resource-dir)

CMAKE ?= cmake

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PICOLIBC_SRC            ?= $(CURDIR)/thirdparty/picolibc
COMPILERRT_BUILTINS_SRC ?= $(CURDIR)/thirdparty/llvm-project/compiler-rt/lib/builtins
LWIP_BUILD_SRC          ?= $(CURDIR)/thirdparty/lwip-build
SYSROOT_OUT             ?= $(CURDIR)/sysroots
BUILD_DIR               ?= $(CURDIR)/build

# ---------------------------------------------------------------------
# Common flags (size-optimized bare-metal defaults)
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
	-nostdlib

# ---------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------
ALL_SYSROOT_TARGETS :=

# ---------------------------------------------------------------------
# Linker flag derivation
#
# -fuse-ld=ld is harmless but noisy; only emit -fuse-ld when LD is
# something other than the default.
# ---------------------------------------------------------------------
ifneq ($(SYSROOT_LD),)
ifneq ($(SYSROOT_LD),ld)
	_SYSROOT_FUSELD := -fuse-ld=$(SYSROOT_LD)
endif
endif

# ---------------------------------------------------------------------
# Toolchain file generator
#
# Compiler-rt isolation flags baked into CMAKE_*_FLAGS_INIT (which
# applies to try_compile probes too):
#
#   --rtlib=compiler-rt    "if you need builtins, use compiler-rt"
#   -resource-dir=<our>    points clang at our resource dir, where
#                          we install our libclang_rt.builtins.a AND
#                          symlink clang's include/ tree
#
# Linker selection flows through CMAKE_*_FLAGS_INIT via
# $(_SYSROOT_FUSELD) -- this also reaches try_compile probes that link.
#
# Args:
#   $(1) = output path
#   $(2) = clang triple
#   $(3) = cmake system processor
#   $(4) = arch CFLAGS (e.g. -mthumb -march=armv7em+fp)
#   $(5) = sysroot path
# ---------------------------------------------------------------------
define generate_toolchain
	@mkdir -p $(dir $(1))
	@printf "\
set(CMAKE_SYSTEM_NAME Generic)\n\
set(CMAKE_SYSTEM_PROCESSOR $(3))\n\
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)\n\
set(CMAKE_C_COMPILER_WORKS 1)\n\
set(CMAKE_CXX_COMPILER_WORKS 1)\n\
set(CMAKE_ASM_COMPILER_WORKS 1)\n\
set(CMAKE_C_COMPILER $(SYSROOT_CC))\n\
set(CMAKE_CXX_COMPILER $(SYSROOT_CXX))\n\
set(CMAKE_ASM_COMPILER $(SYSROOT_CC))\n\
set(CMAKE_C_COMPILER_TARGET $(2))\n\
set(CMAKE_CXX_COMPILER_TARGET $(2))\n\
set(CMAKE_ASM_COMPILER_TARGET $(2))\n\
set(CMAKE_AR $(SYSROOT_AR) CACHE FILEPATH \"\")\n\
set(CMAKE_NM $(SYSROOT_NM) CACHE FILEPATH \"\")\n\
set(CMAKE_RANLIB $(SYSROOT_RANLIB) CACHE FILEPATH \"\")\n\
set(CMAKE_STRIP $(SYSROOT_STRIP) CACHE FILEPATH \"\")\n\
set(CMAKE_SYSROOT $(5))\n\
set(CMAKE_FIND_ROOT_PATH $(5))\n\
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)\n\
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)\n\
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)\n\
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)\n\
set(_sigo_arch_flags \"$(4) --rtlib=compiler-rt -resource-dir=$(5)/lib/clang-resource-dir\")\n\
set(_sigo_link_flags \"$(_SYSROOT_FUSELD)\")\n\
set(CMAKE_C_FLAGS_INIT \"\$${_sigo_arch_flags}\")\n\
set(CMAKE_CXX_FLAGS_INIT \"\$${_sigo_arch_flags}\")\n\
set(CMAKE_ASM_FLAGS_INIT \"$(4)\")\n\
set(CMAKE_EXE_LINKER_FLAGS_INIT \"\$${_sigo_link_flags}\")\n\
set(CMAKE_SHARED_LINKER_FLAGS_INIT \"\$${_sigo_link_flags}\")\n\
set(CMAKE_MODULE_LINKER_FLAGS_INIT \"\$${_sigo_link_flags}\")\n\
" > $(1)
endef

# ---------------------------------------------------------------------
# build_sysroot macro
#
# Arguments:
#   $(1) = target name           (e.g. armv7em-fp)
#   $(2) = clang --target triple (e.g. armv7em-none-eabi)
#   $(3) = arch-specific CFLAGS  (e.g. -mthumb -march=armv7em+fp)
#   $(4) = cmake system proc     (e.g. arm)
# ---------------------------------------------------------------------
define build_sysroot

# -- Per-target directories ------------------------------------------
$(strip $(1))_BUILD       := $$(BUILD_DIR)/$(strip $(1))
$(strip $(1))_SYSROOT     := $$(SYSROOT_OUT)/$(strip $(1))
$(strip $(1))_PICOLIBC_BD := $$($(strip $(1))_BUILD)/picolibc
$(strip $(1))_CRT_BD      := $$($(strip $(1))_BUILD)/compiler-rt
$(strip $(1))_LWIP_BD     := $$($(strip $(1))_BUILD)/lwip
$(strip $(1))_RES_DIR     := $$($(strip $(1))_SYSROOT)/lib/clang-resource-dir
$(strip $(1))_TC          := $$($(strip $(1))_BUILD)/tc-$(strip $(1)).cmake
$(strip $(1))_TARGET      := $(strip $(2))
$(strip $(1))_ARCHFLAGS   := $(strip $(3))
$(strip $(1))_CMAKEARCH   := $(strip $(4))

# -- Resource dir bootstrap ------------------------------------------
#
# Created BEFORE picolibc compiles, because picolibc needs float.h
# etc. from this resource dir. We symlink clang's include/ subtree
# (compiler-builtin headers) but NOT clang's lib/ subtree -- that's
# where libclang_rt.builtins-*.a lives, and we install our own there
# in the compiler-rt step below.
#
$$($(strip $(1))_RES_DIR)/include:
	@mkdir -p $$(dir $$@)
	ln -sfn $$(SYSROOT_CLANG_RESOURCE_DIR)/include $$@

# -- Toolchain file --------------------------------------------------
$$($(strip $(1))_TC):
	$$(call generate_toolchain,$$@,$(strip $(2)),$(strip $(4)),$(strip $(3)),$$($(strip $(1))_SYSROOT))

# -- picolibc (CMake) -----------------------------------------------
#
# Picolibc's CMake support uses different option names than meson.
# Mapping for the options we care about:
#   meson -Dthread-local-storage=false -> CMake -DPICOLIBC_TLS=OFF
#   meson -Dposix-console=true         -> CMake -DPOSIX_CONSOLE=ON
# (multilib, specsdir, tests, newlib-global-atexit have no equivalent.)
#
# PICOLIBC_TLS=OFF makes errno a plain global rather than a __thread
# variable. This avoids pulling in __aeabi_read_tp / _set_tls, which
# we'd otherwise need to either implement or enable separately
# (_HAVE_PICOLIBC_TLS_API). sigo's goroutines aren't OS threads, so a
# single shared errno is consistent with the rest of the runtime.
#
# Install layout: picolibc CMake honors CMAKE_INSTALL_INCLUDEDIR /
# CMAKE_INSTALL_LIBDIR (via GNUInstallDirs). It may install headers
# into a "picolibc/" subdirectory; we expose top-level symlinks so
# that #include <stdio.h> works without extra -I flags.
#
$$($(strip $(1))_SYSROOT)/lib/libc.a: \
		$$($(strip $(1))_TC) \
		$$($(strip $(1))_RES_DIR)/include
	@echo "---- picolibc [$(strip $(1))] ----"
	$$(CMAKE) $$(PICOLIBC_SRC) \
		-G Ninja \
		-B $$($(strip $(1))_PICOLIBC_BD) \
		-DCMAKE_TOOLCHAIN_FILE=$$($(strip $(1))_TC) \
		-DCMAKE_BUILD_TYPE=MinSizeRel \
		-DCMAKE_INSTALL_PREFIX=$$($(strip $(1))_SYSROOT) \
		-DCMAKE_INSTALL_INCLUDEDIR=include \
		-DCMAKE_INSTALL_LIBDIR=lib \
		-DPICOLIBC_TLS=OFF \
		-DPOSIX_IO=ON \
		-DPOSIX_CONSOLE=ON \
		-DTINY_STDIO=ON \
		-D__IO_FLOAT=ON \
		-DPREFER_SIZE_OVER_SPEED=ON
	$$(CMAKE) --build $$($(strip $(1))_PICOLIBC_BD) --target install
	@# picolibc CMake may install to <prefix>/lib/picolibc/<arch>/.
	@# Locate libc.a and surface it at <prefix>/lib/libc.a.
	@if [ ! -f "$$@" ]; then \
		found=$$$$(find $$($(strip $(1))_SYSROOT) -name 'libc.a' -print -quit); \
		if [ -n "$$$$found" ] && [ "$$$$found" != "$$@" ]; then \
			ln -sf "$$$$found" "$$@"; \
		fi; \
	fi
	@# Ditto for the headers -- if they got installed under a
	@# picolibc/ subdir, expose them at the top of include/ too.
	@if [ -d "$$($(strip $(1))_SYSROOT)/include/picolibc" ] && \
	    [ ! -f "$$($(strip $(1))_SYSROOT)/include/stdio.h" ]; then \
		for h in $$($(strip $(1))_SYSROOT)/include/picolibc/*; do \
			ln -sfn "$$$$h" "$$($(strip $(1))_SYSROOT)/include/$$$$(basename $$$$h)"; \
		done; \
	fi

# -- compiler-rt builtins (CMake) -----------------------------------
#
# Installs into $SYSROOT/lib/, then we replicate the archive into our
# clang-resource-dir/lib/<triple>/ so clang's automatic builtins
# lookup finds it. Files installed under both filename layouts that
# clang has used over the years (triple/ and baremetal/).
#
# LLVM_RUNTIMES_BUILD=ON suppresses load_llvm_config() which would
# otherwise pull in host LLVM CMake targets.
#
$$($(strip $(1))_SYSROOT)/lib/libclang_rt.builtins.a: \
		$$($(strip $(1))_TC) \
		$$($(strip $(1))_SYSROOT)/lib/libc.a \
		$$($(strip $(1))_RES_DIR)/include
	@echo "---- compiler-rt [$(strip $(1))] ----"
	$$(CMAKE) $$(COMPILERRT_BUILTINS_SRC) \
		-G Ninja \
		-B $$($(strip $(1))_CRT_BD) \
		-DCMAKE_TOOLCHAIN_FILE=$$($(strip $(1))_TC) \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_INSTALL_PREFIX=$$($(strip $(1))_SYSROOT) \
		-DBUILD_SHARED_LIBS=OFF \
		-DLLVM_RUNTIMES_BUILD=ON \
		-DCOMPILER_RT_OS_DIR="$(strip $(2))" \
		-DCOMPILER_RT_DEFAULT_TARGET_ONLY=ON \
		-DCOMPILER_RT_BAREMETAL_BUILD=ON
	$$(CMAKE) --build $$($(strip $(1))_CRT_BD) --target install
	@installed=$$$$(find $$($(strip $(1))_SYSROOT) -name 'libclang_rt.builtins*.a' -print -quit); \
	if [ -z "$$$$installed" ]; then \
		echo "ERROR: compiler-rt built no library for $(strip $(1))"; \
		exit 1; \
	fi; \
	echo "---- compiler-rt installed: $$$$installed ----"; \
	rm -f $$@; cp "$$$$installed" $$@; \
	resdir=$$($(strip $(1))_RES_DIR); \
	mkdir -p $$$$resdir/lib/$(strip $(2)); \
	mkdir -p $$$$resdir/lib/baremetal; \
	cp "$$$$installed" $$$$resdir/lib/$(strip $(2))/libclang_rt.builtins.a; \
	arch=$$$$(echo $(strip $(2)) | cut -d- -f1); \
	cp "$$$$installed" $$$$resdir/lib/baremetal/libclang_rt.builtins-$$$$arch.a; \
	cp "$$$$installed" $$$$resdir/lib/$(strip $(2))/libclang_rt.builtins-$$$$arch.a

# -- lwip (CMake) ---------------------------------------------------
#
# Depends on compiler-rt being installed first so that any try_compile
# checks (or actual link steps) inside lwIP's CMakeLists pick up our
# builtins via -resource-dir, not the chosen toolchain's bundled ones.
#
$$($(strip $(1))_SYSROOT)/lib/liblwip.a: \
		$$($(strip $(1))_TC) \
		$$($(strip $(1))_SYSROOT)/lib/libc.a \
		$$($(strip $(1))_SYSROOT)/lib/libclang_rt.builtins.a
	@echo "---- lwip [$(strip $(1))] ----"
	$$(CMAKE) $$(LWIP_BUILD_SRC) \
		-G Ninja \
		-B $$($(strip $(1))_LWIP_BD) \
		-DCMAKE_TOOLCHAIN_FILE=$$($(strip $(1))_TC) \
		-DCMAKE_BUILD_TYPE=MinSizeRel \
		-DCMAKE_INSTALL_PREFIX=$$($(strip $(1))_SYSROOT) \
		-DCMAKE_C_FLAGS="$$(COMMON_CFLAGS)"
	$$(CMAKE) --build $$($(strip $(1))_LWIP_BD) --target install
	cp $$(LWIP_BUILD_SRC)/lwipopts.h $$($(strip $(1))_SYSROOT)/include/lwipopts.h

# -- Per-target phony ------------------------------------------------
.PHONY: sysroot-$(strip $(1)) clean-sysroot-$(strip $(1))

sysroot-$(strip $(1)): \
	$$($(strip $(1))_SYSROOT)/lib/libc.a \
	$$($(strip $(1))_SYSROOT)/lib/libclang_rt.builtins.a \
	$$($(strip $(1))_SYSROOT)/lib/liblwip.a
	@echo "---- sysroot ready: $$($(strip $(1))_SYSROOT) ----"

clean-sysroot-$(strip $(1)):
	rm -rf $$($(strip $(1))_BUILD) $$($(strip $(1))_SYSROOT)

ALL_SYSROOT_TARGETS += sysroot-$(strip $(1))

endef  # build_sysroot


# ---------------------------------------------------------------------
# Register targets
# ---------------------------------------------------------------------
#                      name                   clang-target              cflags                                cmake-arch
$(eval $(call build_sysroot,armv6m-nofp,      armv6m-none-eabi,         -mthumb -march=armv6m+nofp,           arm))
$(eval $(call build_sysroot,armv7m-nofp,      armv7m-none-eabi,         -mthumb -march=armv7m+nofp,           arm))
$(eval $(call build_sysroot,armv7em-nofp,     armv7em-none-eabi,        -mthumb -march=armv7em+nofp,          arm))
$(eval $(call build_sysroot,armv7em-fp,       armv7em-none-eabi,        -mthumb -march=armv7em+fp,            arm))
$(eval $(call build_sysroot,armv8m.base-nofp, armv8m.base-none-eabi,    -mthumb -march=armv8m.base+nofp,      arm))
$(eval $(call build_sysroot,armv8m.main-fp,   armv8m.main-none-eabi,    -mthumb -march=armv8m.main+fp,        arm))
#$(eval $(call build_sysroot,armv81m.main-fp, armv81m.main-none-eabi,   -mthumb -march=armv8.1m.main+fp,      arm))
#$(eval $(call build_sysroot,riscv32-imac,    riscv32-none-elf,         -march=rv32imac -mabi=ilp32,          riscv32))

# ---------------------------------------------------------------------
# Aggregate targets
# ---------------------------------------------------------------------
.PHONY: sysroots clean-sysroots print-sysroot-toolchain

sysroots: $(ALL_SYSROOT_TARGETS)

clean-sysroots:
	rm -rf $(BUILD_DIR) $(SYSROOT_OUT)

print-sysroot-toolchain:
	@echo "SYSROOT_CC                 = $(SYSROOT_CC)"
	@echo "SYSROOT_CXX                = $(SYSROOT_CXX)"
	@echo "SYSROOT_LD                 = $(SYSROOT_LD)"
	@echo "SYSROOT_AR                 = $(SYSROOT_AR)"
	@echo "SYSROOT_NM                 = $(SYSROOT_NM)"
	@echo "SYSROOT_RANLIB             = $(SYSROOT_RANLIB)"
	@echo "SYSROOT_STRIP              = $(SYSROOT_STRIP)"
	@echo "SYSROOT_CLANG_RESOURCE_DIR = $(SYSROOT_CLANG_RESOURCE_DIR)"
	@echo "_SYSROOT_FUSELD            = $(_SYSROOT_FUSELD)"