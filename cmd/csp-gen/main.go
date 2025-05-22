package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"

	"pkg.si-go.dev/sigo/targets/device"
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
	// Create the output directory
	if err := os.MkdirAll(outputDir, os.ModePerm); err != nil {
		log.Fatal("file io error: ", err)
	}

	// Open the input file
	file, err := os.Open(input)
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

	// Unmarshal json.
	var d device.Device
	err = json.Unmarshal(buf, &d)
	if err != nil {
		log.Fatal("error decoding json: ", err)
	}
	d.Finalize()

	// Write peripherals API.
	for _, p := range d.Peripherals {
		outFile := outputDir

		if len(p.Group) > 0 {
			// Place packages for grouped peripherals under a common subdirectory.
			outFile = filepath.Join(outFile, formatSymbol(p.Group, false))
		}
		outFile = filepath.Join(outFile, formatSymbol(p.Identifier, false), "peripheral.go")

		// Create the directory structure for the group.
		if err = os.MkdirAll(filepath.Dir(outFile), os.ModePerm); err != nil {
			log.Fatal("file io error: ", err)
		}

		// Create the file.
		f, err := os.OpenFile(outFile, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
		if err != nil {
			log.Fatal(err)
		}

		// Write the peripherals API to the file.
		if _, err := writeBuildTags(f, d, p.BuildTags); err != nil {
			log.Fatal(err)
		}

		if _, err = writePeripheralsApi(f, p); err != nil {
			log.Fatal(err)
		}
	}

	// Write interrupts API.
	for _, v := range d.Variants {
		outFile := filepath.Join(outputDir, "..", fmt.Sprintf("interrupts_%s.go", v.Identifier))

		// Create the directory structure for the group.
		if err = os.MkdirAll(filepath.Dir(outFile), os.ModePerm); err != nil {
			log.Fatal("file io error: ", err)
		}

		// Create the file.
		f, err := os.OpenFile(outFile, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
		if err != nil {
			log.Fatal(err)
		}

		// Write the interrupts API to the file.
		if _, err := writeBuildTags(f, d, nil); err != nil {
			log.Fatal(err)
		}

		// TODO: Support other architectures.
		if _, err = writeInterruptsApi(f, d.Series, "runtime/arm/cortexm", "cortexm.Interrupt", v); err != nil {
			log.Fatal(err)
		}
	}

	// Write the ISR vectors.
	for _, v := range d.Variants {
		outFile := filepath.Join(outputDir, fmt.Sprintf("isr_%s.s", v.Identifier))

		if err = os.MkdirAll(filepath.Dir(outFile), os.ModePerm); err != nil {
			log.Fatal("file io error: ", err)
		}

		// Create the file.
		f, err := os.OpenFile(outFile, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
		if err != nil {
			log.Fatal(err)
		}

		// Write the ISR vector to the file.
		if _, err = writeIsrVector(f, v); err != nil {
			log.Fatal(err)
		}
	}

	// Write the linker scripts.
	for _, v := range d.Variants {
		outFile := filepath.Join(outputDir, fmt.Sprintf("linker_%s.ld", v.Identifier))

		if err = os.MkdirAll(filepath.Dir(outFile), os.ModePerm); err != nil {
			log.Fatal("file io error: ", err)
		}

		// Create the file.
		f, err := os.OpenFile(outFile, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
		if err != nil {
			log.Fatal(err)
		}

		// Write the ISR vector to the file.
		if _, err = writeLinkerScript(f, v); err != nil {
			log.Fatal(err)
		}
	}
}
