# SiGo
A Go compiler and language implementation for embedded systems

NOTE: This compiler is under active development and will change often!

## Supported Architectures

+ ARM Cortex-M0
+ ARM Cortex-M4

Others coming soon...

## Building

NOTE: The Go compiler and Clang is required on PATH to build!

Run the following commands to create a debug build:
```shell
git clone https://github.com/waj334/sigo.git --recurse-submodules
cd ./sigo
make sigo
make generate-csp
make build-picolibc
make build-compiler-rt
./bin/sigoc --help
```

or release:
```shell
make release SIGO_BUILD_RELEASE=1
```

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
    so Go will invoke it instead. Linking will be 100 times faster!
