# SiGo

A Go compiler and language implementation for embedded systems.

> **Note**: This compiler is under active development and changes often.

## Supported architectures

- ARM Cortex-M0+
- ARM Cortex-M4
- ARM Cortex-M7

More to come.

## Prerequisites

- **Go** 1.21 or later
- **Clang** — recommended over GCC; cross-compiles to all supported targets out of the box. On Windows, only the `x86_64-pc-windows-gnu` variant of clang is compatible with Go.
- **CMake** and **Ninja** — for building the LLVM/MLIR submodule and the GoIR dialect
- **System libraries** — on Debian/Ubuntu:
  ```sh
  sudo apt install clang-20 lld-20 cmake ninja-build libzstd-dev
  ```

> **Heads-up on `clang-21`**: as of this writing, `clang-21` crashes when compiling parts of MLIR. Stick with `clang-20` (or GCC) as the host compiler for building LLVM. This is a clang frontend bug, not a SiGo issue.

## Building SiGo

The build has three phases. The first is one-time; the latter two iterate while you develop.

### 1. Build LLVM (one-time, ~30 minutes)

LLVM lives at a fixed location independent of SiGo's build mode and is always built Release. You only rebuild it when the submodule moves.

```sh
git clone https://github.com/waj334/sigo.git --recurse-submodules
cd sigo
make CC=clang-20 CXX=clang++-20 LD=lld install-llvm
```

This installs LLVM to `build/llvm/install`. To use a faster linker for this step (highly recommended — links 10–20x faster than GNU `ld`):

```sh
make CC=clang-20 CXX=clang++-20 LD=lld install-llvm    # LLVM's lld
make CC=clang-20 CXX=clang++-20 LD=mold install-llvm   # mold, often even faster
```

### 2. Build sigoc

```sh
make sigo
```

This builds GoIR, the sysroots (picolibc + compiler-rt + lwIP for each target architecture), and finally the `sigoc` binary at `./bin/sigoc`. A debug build is the default; for a release build:

```sh
make SIGO_BUILD_RELEASE=1 sigo
```

GoIR and sigoc are rebuilt in the same mode (Debug or Release), but LLVM is shared between them. Switching modes does not trigger an LLVM rebuild.

### 3. Verify

```sh
./bin/sigoc --help
```

## Toolchain selection

Three variables control which compiler and linker the build uses:

| Variable | Purpose | Default |
|---|---|---|
| `CC`  | Host C compiler   | system `cc` |
| `CXX` | Host C++ compiler | system `c++` |
| `LD`  | Linker name passed to `-fuse-ld` | `ld` (system default) |

`LD` accepts short names (`lld`, `mold`, `gold`), versioned names (`lld-20`, `ld.lld-20`), or absolute paths (`/usr/bin/ld.lld-20`). All three forms are normalized internally.

```sh
# Versioned toolchain
make CC=clang-20 CXX=clang++-20 LD=lld-20 install-llvm sigo

# Absolute paths if needed
make CC=/opt/llvm-20/bin/clang \
     CXX=/opt/llvm-20/bin/clang++ \
     LD=/opt/llvm-20/bin/ld.lld \
     install-llvm sigo
```

If these are omitted, CMake picks the host system default.

### Sysroot toolchain

The sysroots are cross-compiled to bare-metal ARM/RISC-V targets, so they need a clang (GCC isn't a cross-compiler in the same way). By default, the sysroot build follows `CC` if it's clang-flavored, otherwise it falls back to `which clang`. You can override independently:

```sh
make SYSROOT_CC=clang-20 SYSROOT_CXX=clang++-20 SYSROOT_LD=lld-20 sysroots
```

## Using a pre-installed LLVM

If you already have LLVM/MLIR built somewhere, point at it instead of building one:

```sh
make LLVM_PREFIX=/usr/local sigo
```

This skips `install-llvm` entirely. Run `make env` first to confirm what got resolved.

## Available targets

```
install-llvm        Build and install LLVM (one-time)
sigo                Build sigoc (default target)
release             Same as `sigo` but for release builds
sysroots            Build all sysroots (libc, compiler-rt, lwIP per target)
test                Run SSA compiler tests
env                 Print resolved build flags (useful for debugging)
clean               Remove all build artifacts including LLVM
clean-sigo          Remove only the sigoc binary
clean-sysroots      Remove all sysroots
clean-goir          Remove the GoIR build tree
clean-llvm          Remove the LLVM build tree (forces a full rebuild)
configure           Configure both LLVM and GoIR CMake builds
reconfigure         Force re-configure (useful after toolchain changes)
```

## Related Projects

### Device Drivers and Chip Support

- **[https://github.com/waj334/chip](https://github.com/waj334/chip)**: This repository contains the chip definition files and low-level hardware abstractions used by SiGo.
- **[https://github.com/waj334/drivers](https://github.com/waj334/drivers)**: A collection of device drivers written in Go for various peripherals, designed to work with the SiGo compiler and the `chip` repository.

### Example Firmware

- **[https://github.com/waj334/sigo-arduino-giga-r1-example](https://github.com/waj334/sigo-arduino-giga-r1-example)**: An example firmware for the Arduino GIGA R1, demonstrating how to use SiGo to build a complete application.

## Compiling a firmware image

A firmware image is built with the `build` subcommand. A target CPU must be specified via `--cpu`:

```sh
./bin/sigoc build -j8 --cpu atsame51g19a -O0 -g \
    -o /path/to/output/firmware.elf \
    ./examples/arm/samx51/blinky
```

Full flag reference:

```
Usage:
  sigoc build [flags]

Flags:
      --cpu string       target cpu
      --ctypenames       use C type names for primitives in debug information
  -g, --debug            generate debug information
      --dump-ir          dump the IR
      --float string     floating-point mode (softfp, hardfp) (default "softfp")
  -h, --help             help for build
  -j, --jobs int         number of concurrent builds (default NCPU)
  -O, --opt string       optimization level (default "0")
  -o, --output string    output file (default ".")
  -s, --stack-size int   stack size of each goroutine (default 2048)
  -t, --tags string      build tags
  -v, --verbose string   verbosity level
      --work             do not delete the work directory upon build
```

## Troubleshooting

**`make sigo` fails immediately with "LLVM not found"** — Run `make install-llvm` first. The error message tells you which variable would let you point at an existing LLVM install instead.

**Linking takes forever** — Add `LD=lld` (or `LD=mold`) to your make invocation. Default GNU `ld` can take minutes to link sigoc; `lld` brings it down to seconds.

**`clang-21` crashes during the LLVM build** — Known issue. Use `clang-20` or GCC for the host build (`make CC=clang-20 CXX=clang++-20 install-llvm`).

**`-fuse-ld=ld.lld-NN` rejected as "invalid linker name"** — Should be fixed in the current Makefile, but if you hit it: clang only accepts short suffixes for `-fuse-ld`. The Makefile auto-strips `ld.` prefixes, so `LD=ld.lld-22` becomes `-fuse-ld=lld-22` internally. If you're building outside of `make`, pass `-fuse-ld=lld-22`, not `-fuse-ld=ld.lld-22`.

**Windows: symlink errors during sysroot build** — Enable Developer Mode in Windows settings.

**Need to see what flags `make` is actually passing?** — `make env` prints the resolved `CGO_CFLAGS`, `CGO_LDFLAGS`, etc., plus runs `go env` with them set.

## License

See `LICENSE`.