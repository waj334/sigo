package builder

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"sync"

	"omibyte.io/sigo/targets"
)

const (
	picolibcH = `
#pragma once
#define ATOMIC_UNGETC
#define FAST_STRCMP
#define POSIX_IO
#define TINY_STDIO
#define _HAVE_ALIAS_ATTRIBUTE
#define _HAVE_ALLOC_SIZE
#define _HAVE_ATTRIBUTE_ALWAYS_INLINE
#define _HAVE_ATTRIBUTE_GNU_INLINE
#define _HAVE_BITFIELDS_IN_PACKED_STRUCTS
#define _HAVE_BUILTIN_ALLOCA
#define _HAVE_BUILTIN_COPYSIGN
#define _HAVE_BUILTIN_COPYSIGNL
#define _HAVE_BUILTIN_CTZ
#define _HAVE_BUILTIN_CTZL
#define _HAVE_BUILTIN_CTZLL
#define _HAVE_BUILTIN_EXPECT
#define _HAVE_BUILTIN_FFS
#define _HAVE_BUILTIN_FFSL
#define _HAVE_BUILTIN_FFSLL
#define _HAVE_BUILTIN_ISFINITE
#define _HAVE_BUILTIN_ISINF
#define _HAVE_BUILTIN_ISNAN
#define _HAVE_BUILTIN_MUL_OVERFLOW
#define _HAVE_BUILTIN_ADD_OVERFLOW
#define _HAVE_COMPLEX
#define _HAVE_BUILTIN_COMPLEX
#define _HAVE_FORMAT_ATTRIBUTE
#define _HAVE_INITFINI_ARRAY
#define _HAVE_INIT_FINI
#define _HAVE_LONG_DOUBLE
#define _HAVE_NO_BUILTIN_ATTRIBUTE
#define _HAVE_SEMIHOST
#define _HAVE_WEAK_ATTRIBUTE
#define _ICONV_FROM_ENCODING_
#define _ICONV_FROM_ENCODING_BIG5
#define _ICONV_FROM_ENCODING_CP775
#define _ICONV_FROM_ENCODING_CP850
#define _ICONV_FROM_ENCODING_CP852
#define _ICONV_FROM_ENCODING_CP855
#define _ICONV_FROM_ENCODING_CP866
#define _ICONV_FROM_ENCODING_EUC_JP
#define _ICONV_FROM_ENCODING_EUC_KR
#define _ICONV_FROM_ENCODING_EUC_TW
#define _ICONV_FROM_ENCODING_ISO_8859_1
#define _ICONV_FROM_ENCODING_ISO_8859_10
#define _ICONV_FROM_ENCODING_ISO_8859_11
#define _ICONV_FROM_ENCODING_ISO_8859_13
#define _ICONV_FROM_ENCODING_ISO_8859_14
#define _ICONV_FROM_ENCODING_ISO_8859_15
#define _ICONV_FROM_ENCODING_ISO_8859_2
#define _ICONV_FROM_ENCODING_ISO_8859_3
#define _ICONV_FROM_ENCODING_ISO_8859_4
#define _ICONV_FROM_ENCODING_ISO_8859_5
#define _ICONV_FROM_ENCODING_ISO_8859_6
#define _ICONV_FROM_ENCODING_ISO_8859_7
#define _ICONV_FROM_ENCODING_ISO_8859_8
#define _ICONV_FROM_ENCODING_ISO_8859_9
#define _ICONV_FROM_ENCODING_ISO_IR_111
#define _ICONV_FROM_ENCODING_KOI8_R
#define _ICONV_FROM_ENCODING_KOI8_RU
#define _ICONV_FROM_ENCODING_KOI8_U
#define _ICONV_FROM_ENCODING_KOI8_UNI
#define _ICONV_FROM_ENCODING_UCS_2
#define _ICONV_FROM_ENCODING_UCS_2BE
#define _ICONV_FROM_ENCODING_UCS_2LE
#define _ICONV_FROM_ENCODING_UCS_2_INTERNAL
#define _ICONV_FROM_ENCODING_UCS_4
#define _ICONV_FROM_ENCODING_UCS_4BE
#define _ICONV_FROM_ENCODING_UCS_4LE
#define _ICONV_FROM_ENCODING_UCS_4_INTERNAL
#define _ICONV_FROM_ENCODING_US_ASCII
#define _ICONV_FROM_ENCODING_UTF_16
#define _ICONV_FROM_ENCODING_UTF_16BE
#define _ICONV_FROM_ENCODING_UTF_16LE
#define _ICONV_FROM_ENCODING_UTF_8
#define _ICONV_FROM_ENCODING_WIN_1250
#define _ICONV_FROM_ENCODING_WIN_1251
#define _ICONV_FROM_ENCODING_WIN_1252
#define _ICONV_FROM_ENCODING_WIN_1253
#define _ICONV_FROM_ENCODING_WIN_1254
#define _ICONV_FROM_ENCODING_WIN_1255
#define _ICONV_FROM_ENCODING_WIN_1256
#define _ICONV_FROM_ENCODING_WIN_1257
#define _ICONV_FROM_ENCODING_WIN_1258
#define _ICONV_TO_ENCODING_
#define _ICONV_TO_ENCODING_BIG5
#define _ICONV_TO_ENCODING_CP775
#define _ICONV_TO_ENCODING_CP850
#define _ICONV_TO_ENCODING_CP852
#define _ICONV_TO_ENCODING_CP855
#define _ICONV_TO_ENCODING_CP866
#define _ICONV_TO_ENCODING_EUC_JP
#define _ICONV_TO_ENCODING_EUC_KR
#define _ICONV_TO_ENCODING_EUC_TW
#define _ICONV_TO_ENCODING_ISO_8859_1
#define _ICONV_TO_ENCODING_ISO_8859_10
#define _ICONV_TO_ENCODING_ISO_8859_11
#define _ICONV_TO_ENCODING_ISO_8859_13
#define _ICONV_TO_ENCODING_ISO_8859_14
#define _ICONV_TO_ENCODING_ISO_8859_15
#define _ICONV_TO_ENCODING_ISO_8859_2
#define _ICONV_TO_ENCODING_ISO_8859_3
#define _ICONV_TO_ENCODING_ISO_8859_4
#define _ICONV_TO_ENCODING_ISO_8859_5
#define _ICONV_TO_ENCODING_ISO_8859_6
#define _ICONV_TO_ENCODING_ISO_8859_7
#define _ICONV_TO_ENCODING_ISO_8859_8
#define _ICONV_TO_ENCODING_ISO_8859_9
#define _ICONV_TO_ENCODING_ISO_IR_111
#define _ICONV_TO_ENCODING_KOI8_R
#define _ICONV_TO_ENCODING_KOI8_RU
#define _ICONV_TO_ENCODING_KOI8_U
#define _ICONV_TO_ENCODING_KOI8_UNI
#define _ICONV_TO_ENCODING_UCS_2
#define _ICONV_TO_ENCODING_UCS_2BE
#define _ICONV_TO_ENCODING_UCS_2LE
#define _ICONV_TO_ENCODING_UCS_2_INTERNAL
#define _ICONV_TO_ENCODING_UCS_4
#define _ICONV_TO_ENCODING_UCS_4BE
#define _ICONV_TO_ENCODING_UCS_4LE
#define _ICONV_TO_ENCODING_UCS_4_INTERNAL
#define _ICONV_TO_ENCODING_US_ASCII
#define _ICONV_TO_ENCODING_UTF_16
#define _ICONV_TO_ENCODING_UTF_16BE
#define _ICONV_TO_ENCODING_UTF_16LE
#define _ICONV_TO_ENCODING_UTF_8
#define _ICONV_TO_ENCODING_WIN_1250
#define _ICONV_TO_ENCODING_WIN_1251
#define _ICONV_TO_ENCODING_WIN_1252
#define _ICONV_TO_ENCODING_WIN_1253
#define _ICONV_TO_ENCODING_WIN_1254
#define _ICONV_TO_ENCODING_WIN_1255
#define _ICONV_TO_ENCODING_WIN_1256
#define _ICONV_TO_ENCODING_WIN_1257
#define _ICONV_TO_ENCODING_WIN_1258
#define _IEEE_LIBM
#define _IO_FLOAT_EXACT
#define _LITE_EXIT
#define _MB_LEN_MAX 1
#define _NANO_MALLOC
#define _NEWLIB_VERSION "4.3.0"
#define _PICO_EXIT
#define _PICOLIBC_MINOR__ 8
#define _PICOLIBC_VERSION "1.8.3"
#define _PICOLIBC__ 1
#define _RETARGETABLE_LOCKING
#define _WANT_IO_C99_FORMATS
#define FORMAT_DEFAULT_INTEGER
#define __NEWLIB_MINOR__ 3
#define __NEWLIB_PATCHLEVEL__ 0
#define __NEWLIB__ 4
#define __OBSOLETE_MATH_FLOAT 1
#define __OBSOLETE_MATH_DOUBLE 1
#define __PICOLIBC_MINOR__ 8
#define __PICOLIBC_PATCHLEVEL__ 3
#define __PICOLIBC_VERSION__ "1.8.3"
#define __PICOLIBC__ 1
`
)

type Package struct {
	PathPrefix  string
	Sources     []string
	IncludeDirs []string
	Defines     []string
	Files       map[string]string
}

func (p *Package) Compile(toolchain Toolchain, target targets.TargetInfo, debug bool, opt string, float string,
	numjobs int, outputDir string) ([]string, error) {

	var artifacts []string
	var defines []string

	includes := []string{"-I" + outputDir}

	// Create files.
	for p, content := range p.Files {
		fname := filepath.Join(outputDir, p)
		err := os.MkdirAll(filepath.Dir(fname), os.ModePerm)
		if err != nil {
			return nil, err
		}

		err = os.WriteFile(fname, []byte(content), 0644)
		if err != nil {
			return nil, err
		}
	}

	for _, include := range p.IncludeDirs {
		includes = append(includes, fmt.Sprintf("-I%s", include))
	}

	for _, define := range p.Defines {
		defines = append(defines, fmt.Sprintf("-D%s", define))
	}

	var wg sync.WaitGroup
	var mutex sync.Mutex
	semaphore := make(chan struct{}, numjobs)

	var errs []error

	wg.Add(len(p.Sources))
	go func() {
		for _, src := range p.Sources {
			semaphore <- struct{}{}
			go func() {
				defer wg.Done()
				defer func() {
					// Allow another goroutine to be spun up.
					<-semaphore
				}()

				artifact := filepath.Join(outputDir, filepath.Base(p.PathPrefix), filepath.Dir(src), filepath.Base(src)+".o")

				// Create the directory to where the artifact should be stored.
				err := os.MkdirAll(filepath.Dir(artifact), os.ModePerm)
				if err != nil {
					mutex.Lock()
					errs = append(errs, err)
					mutex.Unlock()

					return
				}

				args := []string{
					fmt.Sprintf("--target=%s", target.Triplet),
				}
				args = append(args, includes...)
				args = append(args, defines...)

				if debug {
					args = append(args, "-g")
				}

				switch opt {
				case "0":
					args = append(args, "-O0")
				case "1":
					args = append(args, "-O1")
				case "2":
					args = append(args, "-O2")
				case "3":
					args = append(args, "-O3")
				case "s":
					args = append(args, "-Os")
				case "z":
					args = append(args, "-Oz")
				case "d":
					args = append(args, "-O0")
				default:
					args = append(args, "-O0")
				}

				if float == "nofp" {
					args = append(args, "-mfloat-abi=softfp")
				} else {
					args = append(args, "-mfloat-abi=hard")

					if len(target.Fpu.Type) > 0 {
						args = append(args, fmt.Sprintf("-mfpu=%s", target.Fpu.Type))
					}
				}

				args = append(args, fmt.Sprintf("-mcpu=%s", target.Cpu))
				args = append(args,
					"-c", "-o", artifact,
					filepath.Join(p.PathPrefix, src),
				)
				cmd := exec.Command(toolchain.CC, args...)
				cmd.Stdout = os.Stdout
				cmd.Stderr = os.Stderr
				cmd.Stdin = os.Stdin
				if err := cmd.Run(); err != nil {
					mutex.Lock()
					fmt.Println()
					fmt.Println("Command failed: ", cmd.String())
					errs = append(errs, errors.Join(ErrCompilerFailed, err))
					mutex.Unlock()
				}

				mutex.Lock()
				artifacts = append(artifacts, artifact)
				mutex.Unlock()
			}()
		}
	}()

	// Wait for work queue to complete.
	wg.Wait()

	if len(errs) > 0 {
		return nil, errors.Join(errs...)
	}

	return artifacts, nil
}

func pkgPicolibc(target targets.TargetInfo) (Package, error) {
	env, err := Environment()
	if err != nil {
		return Package{}, err
	}

	prefix := filepath.Join(env["SIGOROOT"], "thirdparty", "picolibc")
	pkg := Package{
		PathPrefix: prefix,
		IncludeDirs: []string{
			filepath.Join(prefix, "newlib/libc/include"),
			filepath.Join(prefix, "newlib/libc/stdio"),
			filepath.Join(prefix, "picocrt"),
		},
		Files: map[string]string{
			"picolibc.h": picolibcH,
		},
		Sources: []string{
			"newlib/libc/errno/errno.c",
			"newlib/libc/misc/init.c",
			"newlib/libc/misc/lock.c",
			"newlib/libc/picolib/picosbrk.c",
			"newlib/libc/stdlib/abort.c",
			"newlib/libc/stdlib/exit.c",
			"newlib/libc/stdlib/nano-malloc-calloc.c",
			"newlib/libc/stdlib/nano-malloc-cfree.c",
			"newlib/libc/stdlib/nano-malloc-free.c",
			"newlib/libc/stdlib/nano-malloc-mallinfo.c",
			"newlib/libc/stdlib/nano-malloc-malloc.c",
			"newlib/libc/stdlib/nano-malloc-malloc_stats.c",
			"newlib/libc/stdlib/nano-malloc-malloc_usable_size.c",
			"newlib/libc/stdlib/nano-malloc-mallopt.c",
			"newlib/libc/stdlib/nano-malloc-memalign.c",
			"newlib/libc/stdlib/nano-malloc-pvalloc.c",
			"newlib/libc/stdlib/nano-malloc-realloc.c",
			"newlib/libc/stdlib/nano-malloc-valloc.c",
			"newlib/libc/string/memcpy.c",
			"newlib/libc/string/strncmp.c",
			"newlib/libc/string/strncpy.c",
		},
	}

	switch target.Architecture {
	case "arm", "thumb":
		pkg.Sources = append(pkg.Sources,
			"newlib/libc/machine/arm/bzero.c",
			"newlib/libc/machine/arm/memchr.S",
			"newlib/libc/machine/arm/memmove.c",
			"newlib/libc/machine/arm/memset.c",
			"newlib/libc/machine/arm/setjmp.S",
			"newlib/libc/machine/arm/strcmp.S",
			"newlib/libc/machine/arm/strcpy.S",
			"newlib/libc/machine/arm/strlen.c",
			"newlib/libc/machine/arm/strlen.S",
		)
	}

	return pkg, nil
}

func pkgCompilerRT(target targets.TargetInfo, float string) (Package, error) {
	env, err := Environment()
	if err != nil {
		return Package{}, err
	}

	prefix := filepath.Join(env["SIGOROOT"], "thirdparty", "llvm-project", "compiler-rt")
	pkg := Package{
		PathPrefix: prefix,
		IncludeDirs: []string{
			filepath.Join(prefix, "include"),
		},
		Defines: []string{
			"VISIBILITY_HIDDEN",
		},
		Files:   map[string]string{},
		Sources: []string{},
	}

	hasFp := false
	fpIsDp := false
	if float != "nofp" {
		if slices.Contains(target.Features, "fpregs") {
			hasFp = true
			switch {
			case slices.Contains(target.Features, "fp-armv8d16"):
				fpIsDp = true
			}
		}
	}

	genericLookup := map[string]struct{}{}
	addSources := func(paths ...string) {
		for _, src := range paths {
			p := filepath.Join("lib", "builtins", src)
			if !slices.Contains(pkg.Sources, p) {
				pkg.Sources = append(pkg.Sources, p)
				baseName := strings.Split(filepath.Base(p), ".")[0]
				genericLookup[baseName] = struct{}{}
			}
		}
	}

	filterGeneric := func(generic []string) (out []string) {
		out = make([]string, 0, len(generic))
		for _, genericSource := range generic {
			baseName := strings.Split(filepath.Base(genericSource), ".")[0]
			if _, ok := genericLookup[baseName]; ok {
				continue
			}
			out = append(out, genericSource)
		}
		return
	}

	genericSources := []string{
		"absvdi2.c",
		"absvsi2.c",
		"absvti2.c",
		"adddf3.c",
		"addsf3.c",
		"addvdi3.c",
		"addvsi3.c",
		"addvti3.c",
		"apple_versioning.c",
		"ashldi3.c",
		"ashlti3.c",
		"ashrdi3.c",
		"ashrti3.c",
		"bswapdi2.c",
		"bswapsi2.c",
		"clzdi2.c",
		"clzsi2.c",
		"clzti2.c",
		"cmpdi2.c",
		"cmpti2.c",
		"comparedf2.c",
		"comparesf2.c",
		"ctzdi2.c",
		"ctzsi2.c",
		"ctzti2.c",
		"divdc3.c",
		"divdf3.c",
		"divdi3.c",
		"divmoddi4.c",
		"divmodsi4.c",
		"divmodti4.c",
		"divsc3.c",
		"divsf3.c",
		"divsi3.c",
		"divti3.c",
		"extendsfdf2.c",
		"extendhfsf2.c",
		"extendhfdf2.c",
		"ffsdi2.c",
		"ffssi2.c",
		"ffsti2.c",
		"fixdfdi.c",
		"fixdfsi.c",
		"fixdfti.c",
		"fixsfdi.c",
		"fixsfsi.c",
		"fixsfti.c",
		"fixunsdfdi.c",
		"fixunsdfsi.c",
		"fixunsdfti.c",
		"fixunssfdi.c",
		"fixunssfsi.c",
		"fixunssfti.c",
		"floatdidf.c",
		"floatdisf.c",
		"floatsidf.c",
		"floatsisf.c",
		"floattidf.c",
		"floattisf.c",
		"floatundidf.c",
		"floatundisf.c",
		"floatunsidf.c",
		"floatunsisf.c",
		"floatuntidf.c",
		"floatuntisf.c",
		"fp_mode.c",
		"int_util.c",
		"lshrdi3.c",
		"lshrti3.c",
		"moddi3.c",
		"modsi3.c",
		"modti3.c",
		"muldc3.c",
		"muldf3.c",
		"muldi3.c",
		"mulodi4.c",
		"mulosi4.c",
		"muloti4.c",
		"mulsc3.c",
		"mulsf3.c",
		"multi3.c",
		"mulvdi3.c",
		"mulvsi3.c",
		"mulvti3.c",
		"negdf2.c",
		"negdi2.c",
		"negsf2.c",
		"negti2.c",
		"negvdi2.c",
		"negvsi2.c",
		"negvti2.c",
		"os_version_check.c",
		"paritydi2.c",
		"paritysi2.c",
		"parityti2.c",
		"popcountdi2.c",
		"popcountsi2.c",
		"popcountti2.c",
		"powidf2.c",
		"powisf2.c",
		"subdf3.c",
		"subsf3.c",
		"subvdi3.c",
		"subvsi3.c",
		"subvti3.c",
		"trampoline_setup.c",
		"truncdfhf2.c",
		"truncdfsf2.c",
		"truncsfhf2.c",
		"ucmpdi2.c",
		"ucmpti2.c",
		"udivdi3.c",
		"udivmoddi4.c",
		"udivmodsi4.c",
		"udivmodti4.c",
		"udivsi3.c",
		"udivti3.c",
		"umoddi3.c",
		"umodsi3.c",
		"umodti3.c",
	}

	armBaseSources := []string{
		"arm/fp_mode.c",
		"arm/bswapdi2.S",
		"arm/bswapsi2.S",
		"arm/clzdi2.S",
		"arm/clzsi2.S",
		"arm/comparesf2.S",
		"arm/divmodsi4.S",
		"arm/divsi3.S",
		"arm/modsi3.S",
		"arm/udivmodsi4.S",
		"arm/udivsi3.S",
		"arm/umodsi3.S",
	}

	armEabiSources := []string{
		"arm/aeabi_cdcmp.S",
		"arm/aeabi_cdcmpeq_check_nan.c",
		"arm/aeabi_cfcmp.S",
		"arm/aeabi_cfcmpeq_check_nan.c",
		"arm/aeabi_dcmp.S",
		"arm/aeabi_div0.c",
		"arm/aeabi_drsub.c",
		"arm/aeabi_fcmp.S",
		"arm/aeabi_frsub.c",
		"arm/aeabi_idivmod.S",
		"arm/aeabi_ldivmod.S",
		//"arm/aeabi_memcmp.S",
		//"arm/aeabi_memcpy.S",
		//"arm/aeabi_memmove.S",
		//"arm/aeabi_memset.S",
		"arm/aeabi_uidivmod.S",
		"arm/aeabi_uldivmod.S",
	}

	armSyncSources := []string{
		"arm/sync_fetch_and_add_4.S",
		"arm/sync_fetch_and_add_8.S",
		"arm/sync_fetch_and_and_4.S",
		"arm/sync_fetch_and_and_8.S",
		"arm/sync_fetch_and_max_4.S",
		"arm/sync_fetch_and_max_8.S",
		"arm/sync_fetch_and_min_4.S",
		"arm/sync_fetch_and_min_8.S",
		"arm/sync_fetch_and_nand_4.S",
		"arm/sync_fetch_and_nand_8.S",
		"arm/sync_fetch_and_or_4.S",
		"arm/sync_fetch_and_or_8.S",
		"arm/sync_fetch_and_sub_4.S",
		"arm/sync_fetch_and_sub_8.S",
		"arm/sync_fetch_and_umax_4.S",
		"arm/sync_fetch_and_umax_8.S",
		"arm/sync_fetch_and_umin_4.S",
		"arm/sync_fetch_and_umin_8.S",
		"arm/sync_fetch_and_xor_4.S",
		"arm/sync_fetch_and_xor_8.S",
	}

	thumb1BaseSources := []string{
		"arm/divsi3.S",
		"arm/udivsi3.S",
		"arm/comparesf2.S",
		"arm/addsf3.S",
	}

	thumb1JtSources := []string{
		"arm/switch16.S",
		"arm/switch32.S",
		"arm/switch8.S",
		"arm/switchu8.S",
	}

	thumb1SjLjEhSources := []string{
		"arm/restore_vfp_d8_d15_regs.S",
		"arm/save_vfp_d8_d15_regs.S",
	}

	thumb1Vfp2Dpsources := []string{
		"arm/adddf3vfp.S",
		"arm/divdf3vfp.S",
		"arm/eqdf2vfp.S",
		"arm/extendsfdf2vfp.S",
		"arm/fixdfsivfp.S",
		"arm/fixunsdfsivfp.S",
		"arm/floatsidfvfp.S",
		"arm/floatunssidfvfp.S",
		"arm/gedf2vfp.S",
		"arm/gtdf2vfp.S",
		"arm/ledf2vfp.S",
		"arm/ltdf2vfp.S",
		"arm/muldf3vfp.S",
		"arm/nedf2vfp.S",
		"arm/negdf2vfp.S",
		"arm/subdf3vfp.S",
		"arm/truncdfsf2vfp.S",
		"arm/unorddf2vfp.S",
	}

	thumb1Vfp2Spsources := []string{
		"arm/addsf3vfp.S",
		"arm/divsf3vfp.S",
		"arm/eqsf2vfp.S",
		"arm/fixsfsivfp.S",
		"arm/fixunssfsivfp.S",
		"arm/floatsisfvfp.S",
		"arm/floatunssisfvfp.S",
		"arm/gesf2vfp.S",
		"arm/gtsf2vfp.S",
		"arm/lesf2vfp.S",
		"arm/ltsf2vfp.S",
		"arm/mulsf3vfp.S",
		"arm/negsf2vfp.S",
		"arm/nesf2vfp.S",
		"arm/subsf3vfp.S",
		"arm/unordsf2vfp.S",
	}

	thumb1IcacheSources := []string{
		"arm/sync_synchronize.S",
	}

	// Handle adding target-specific sources.
	triplet := strings.Split(target.Triplet, "-")
	switch triplet[0] {
	case "arm", "armv6m", "armv8m.base":
		addSources(armEabiSources...)
		addSources(thumb1BaseSources...)
	case "armhf", "armv7", "armv7s", "armv7k", "armv7m", "armv7em", "armv8m.main", "armv8.1m.main":
		addSources(armBaseSources...)
		addSources(armSyncSources...)
		addSources(armEabiSources...)
		addSources(thumb1BaseSources...)
		addSources(thumb1JtSources...)
		addSources(thumb1IcacheSources...)

		if hasFp {
			addSources(thumb1SjLjEhSources...)
			if fpIsDp {
				addSources(thumb1Vfp2Dpsources...)
			} else {
				addSources(thumb1Vfp2Spsources...)
			}
		}
	}

	// Filter the generic sources to remove anything that has a target-specific implementation.
	filteredGeneric := filterGeneric(genericSources)

	// Add the filtered set.
	addSources(filteredGeneric...)

	return pkg, nil
}
