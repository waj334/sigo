# SiGo
A Go compiler and language implementation for embedded systems

NOTE: This compiler is under active development and will change often!

## Supported Architectures

+ ARM Cortex-M0+
+ ARM Cortex-M4
+ ARM Cortex-M7

Others coming soon...

## Compiling

### System Requirements:
1. The Go Compiler
2. Clang (Recommended)
   1. On Windows, only the `x86_64-pc-windows-gnu` variant of clang is compatible with Go.

```shell
sudo apt install clang-20 libzstd-dev
```

Run the following commands to create a debug build:
```shell
git clone https://github.com/waj334/sigo.git --recurse-submodules
cd ./sigo

export CC=clang
export CXX=clang++
export LD=ld.lld

make sigo
./bin/sigoc --help
```

or release:
```shell
export CC=clang
export CXX=clang++
export LD=ld.lld

make release SIGO_BUILD_RELEASE=1
```

### Selecting a toolchain

The toolchain used for compilation can be specified like the following:
```shell
export CC=clang-20
export CXX=clang++-20
export LD=ld.lld-20

make sigo
make release SIGO_BUILD_RELEASE=1
```

Clang is the recommended build toolchain since it can cross compile for most target platforms. The GNU compiler suite 
can be used but the correct variant of it must be selected when compiling the runtime libraries for each target 
platform.

**Note: If these values are omitted, then the default toolchain for the host system will be selected by CMake.**

## Compiling a firmware image
A firmware image can be compiled using the `build` subcommand:
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

NOTE: A target CPU must be specified via the `--cpu` flag!

```shell
sigoc build -j8 --cpu atsame51g19a -O0 -g -o/path/to/output/firmware.elf ./examples/arm/samx51/blinky
```

## Troubleshooting

### Windows

+ Enable "Developer Mode" to fix issues with creating symlinks during build directory staging.
+ Slow linking?
  + The Go compiler will use `ld` by default, but `ld.lld` can be renamed to `ld` (rename ld and copy ld.lld in its place) 
    so Go will invoke it instead and linking the final application will be exponentially times faster!
