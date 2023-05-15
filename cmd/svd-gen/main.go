package main

import (
	"encoding/xml"
	"flag"
	"fmt"
	"io"
	"log"
	"omibyte.io/sigo/cmd/svd-gen/generator"
	"omibyte.io/sigo/cmd/svd-gen/generator/SAM"
	"os"

	"omibyte.io/sigo/cmd/svd-gen/svd"
)

var (
	svdIn     string
	outputDir string
)

func init() {
	flag.StringVar(&svdIn, "in", "", "input SVD file")
	flag.StringVar(&outputDir, "out", "", "output directory")
	flag.Parse()
}

func main() {

	// Open the input file
	file, err := os.Open(svdIn)
	if err != nil {
		log.Fatal("file io error: ", err)
	}

	// Read the SVD file into a buffer
	buf, err := io.ReadAll(file)
	if err != nil {
		log.Fatal("io error: ", err)
	}

	// Close the file
	if err = file.Close(); err != nil {
		log.Fatal("file io error: ", err)
	}

	// Decode the SVD XML
	var device svd.DeviceElement
	if err = xml.Unmarshal(buf, &device); err != nil {
		log.Fatal("xml decode error: ", err)
	}

	// Begin device implementation
	fmt.Println("Generating the runtime package for the following machine:")
	fmt.Printf("CPU:\t\t%s\n", device.CPU.Name)
	fmt.Printf("Revision:\t%s\n", device.CPU.Revision)
	fmt.Printf("Endian:\t\t%s\n", device.CPU.Endian)
	fmt.Printf("Architecture:\t%v-bit\n", device.BitWidth)
	fmt.Printf("Addressable Width:\t%v-bit\n", device.AddressableWidth)
	fmt.Printf("FPU:\t\t%v\n", device.CPU.FPUPresent)

	var gen generator.Generator
	// Choose the generator based on series
	switch device.Series {
	case "SAMD21", "SAMD51", "SAME51", "SAME70", "SAML11", "SAML22", "SAMR21", "SAMS70", "SAMV70", "SAMV71":
		gen = SAM.NewGenerator(device)
	default:
		panic("unsupported device")
	}

	// Create the output directory
	if err = os.MkdirAll(outputDir, 0750); err != nil {
		log.Fatal("file io error: ", err)
	}

	// Generate the implementation
	if err = gen.Generate(outputDir); err != nil {
		log.Fatal("generator error: ", err)
	}

	fmt.Println("Done.")
}
