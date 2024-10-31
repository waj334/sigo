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

## Troubleshooting

### Windows

+ Enable "Developer Mode" to fix issues with creating symlinks during build directory staging.
+ Slow linking?
  + The Go compiler will use `ld` by default, but `ld.lld` can be renamed to `ld` (rename ld and copy ld.lld in its place) 
    so Go will invoke it instead. Linking will be 100 times faster!