package main

import (
	"encoding/xml"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"

	"omibyte.io/sigo/cmd/csp-gen/atdf"
	"omibyte.io/sigo/cmd/csp-gen/generator"
	sam_atdf "omibyte.io/sigo/cmd/csp-gen/generator/SAM/atdf"
	sam_svd "omibyte.io/sigo/cmd/csp-gen/generator/SAM/svd"
	"omibyte.io/sigo/cmd/csp-gen/svd"
)

var (
	input     string
	outputDir string
)

func init() {
	flag.StringVar(&input, "in", "", "input file")
	flag.StringVar(&outputDir, "out", "", "output directory")
	flag.Parse()
}

func main() {
	fnames, err := filepath.Glob(input)
	if err != nil {
		log.Fatal(err)
	}

	for _, fname := range fnames {
		filetype := strings.ToLower(filepath.Ext(fname))
		if filetype != ".svd" && filetype != ".atdf" {
			log.Fatalf("Unsupported file type %s", filetype)
		}

		// Open the input file
		file, err := os.Open(fname)
		if err != nil {
			log.Fatal("file io error: ", err)
		}

		// Read the input file into a buffer
		buf, err := io.ReadAll(file)
		if err != nil {
			log.Fatal("io error: ", err)
		}

		// Close the file
		if err = file.Close(); err != nil {
			log.Fatal("file io error: ", err)
		}

		var def any
		switch filetype {
		case ".svd":
			def = &svd.DeviceElement{}
		case ".atdf":
			def = &atdf.ATDF{}
		}

		if err = xml.Unmarshal(buf, def); err != nil {
			log.Fatalf("%s: xml decode error: %v", fname, err)
		}

		// Create the output directory
		if err = os.MkdirAll(outputDir, 0750); err != nil {
			log.Fatal("file io error: ", err)
		}

		// TODO: Support multiple input files

		var gen generator.Generator

		// Choose the generator based on the generator and then the series
		switch def := def.(type) {
		case *atdf.ATDF:
			for _, device := range def.Devices.Elements {
				fmt.Printf("CPU:\t\t%s\n", device.Name)
				switch device.Series {
				case "SAMC21", "SAMD21", "SAMD51", "SAME51", "SAME70", "SAML11", "SAML22", "SAMR21", "SAMS70", "SAMV70", "SAMV71":
					gen = sam_atdf.NewGenerator(def, &device)
				default:
					log.Printf("Unsupported device: %s", device.Name)
					continue
				}

				// Generate the implementation
				if err = gen.Generate(outputDir); err != nil {
					log.Fatal("generator error: ", err)
				}
			}
		case *svd.DeviceElement:
			fmt.Println("Generating the runtime package for the following machine:")
			fmt.Printf("CPU:\t\t%s\n", def.CPU.Name)
			fmt.Printf("Revision:\t%s\n", def.CPU.Revision)
			fmt.Printf("Endian:\t\t%s\n", def.CPU.Endian)
			fmt.Printf("Architecture:\t%v-bit\n", def.BitWidth)
			fmt.Printf("Addressable Width:\t%v-bit\n", def.AddressableWidth)
			fmt.Printf("FPU:\t\t%v\n", def.CPU.FPUPresent)

			switch def.Series {
			case "SAMD21", "SAMD51", "SAME51", "SAME70", "SAML11", "SAML22", "SAMR21", "SAMS70", "SAMV70", "SAMV71":
				gen = sam_svd.NewGenerator(def)

				// TODO: Move this when multiple input files are supported
				// Generate the implementation
				if err = gen.Generate(outputDir); err != nil {
					log.Fatal("generator error: ", err)
				}
			default:
				log.Printf("Unsupported device: %s", def.Name)
			}
		}

		fmt.Println("Done.")
	}
}
