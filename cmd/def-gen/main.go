package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"omibyte.io/sigo/targets/device"
	"omibyte.io/sigo/targets/device/importer"
	"omibyte.io/sigo/targets/device/importer/atdf"
	"omibyte.io/sigo/targets/device/importer/svd"
)

var (
	base            string
	variantPatterns string
	output          string
)

func init() {
	flag.StringVar(&base, "base", "", "base description file for series")
	flag.StringVar(&variantPatterns, "variants", "", "description file patterns for series variants")
	flag.StringVar(&output, "output", "", "output SiGO description file")
	flag.Parse()
}

func main() {
	var baseDevice device.Device
	var err error
	ctx := context.Background()

	if len(base) == 0 {
		println("no base definition file specified")
		os.Exit(-1)
	}

	switch filepath.Ext(base) {
	case ".svd":
		baseDevice, err = svd.ImportSVD(ctx, importer.Config{
			BaseFilename: base,
			OnlyVariants: false,
		})
		if err != nil {
			println(err)
			os.Exit(-3)
		}

	case ".atdf":
		baseDevice, err = atdf.ImportATDF(ctx, importer.Config{
			BaseFilename: base,
			OnlyVariants: false,
		})
		if err != nil {
			println(err)
			os.Exit(-3)
		}

		if len(variantPatterns) > 0 {
			var variantFiles []string
			for _, pattern := range strings.Split(variantPatterns, ",") {
				// Prepend the directory of the base file to the pattern.
				pattern = filepath.Join(filepath.Dir(base), pattern)
				variants, err := filepath.Glob(pattern)
				if err != nil {
					println(err)
					os.Exit(-4)
				}
				variantFiles = append(variantFiles, variants...)
			}

			for _, fname := range variantFiles {
				variantDevice, err := atdf.ImportATDF(ctx, importer.Config{
					BaseFilename: fname,
					OnlyVariants: true,
				})

				if err != nil {
					println(err)
					os.Exit(-3)
				}

				// Merge variant information.
				for _, variant := range variantDevice.Variants {
					exists := slices.ContainsFunc(baseDevice.Variants, func(a device.Variant) bool {
						return a.Identifier == variant.Identifier
					})

					if !exists {
						baseDevice.Variants = append(baseDevice.Variants, variant)
					}
				}
			}
		}

	default:
		println("unsupported input definition file type")
		os.Exit(-2)
	}

	b, err := json.MarshalIndent(&baseDevice, "", "  ")
	if err != nil {
		panic(err)
	}

	if len(output) > 0 {
		// Dump to the specified output file.
		f, err := os.OpenFile(output, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
		if err != nil {
			println(err)
			os.Exit(-4)
		}

		if _, err = fmt.Fprintln(f, string(b)); err != nil {
			println(err)
			os.Exit(-4)
		}

		if err = f.Close(); err != nil {
			println(err)
			os.Exit(-4)
		}
	} else {
		// Dump to stdout.
		fmt.Println(string(b))
	}
}
