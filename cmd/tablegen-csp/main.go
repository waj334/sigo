package main

import (
	"flag"
	"fmt"
	"log"
	"omibyte.io/sigo/llvm/tablegen"
)

var (
	input     string
	outputDir string
	includes  includeDirs
)

type includeDirs []string

func (i *includeDirs) String() string {
	return fmt.Sprintf("%v", *i)
}

func (i *includeDirs) Set(value string) error {
	*i = append(*i, value)
	return nil
}

func init() {
	flag.StringVar(&input, "in", "", "input file")
	flag.StringVar(&outputDir, "out", "", "output directory")
	flag.Var(&includes, "I", "Add an include directory")
	flag.Parse()
}

func main() {
	rk := tablegen.NewRecordKeeper()
	ok := tablegen.ParseTableGenFile(input, rk, includes)
	if !ok {
		log.Fatalf("failed to parse the input file")
	}

	for _, class := range rk.GetClasses() {
		fmt.Printf("class: %s\n", class.GetName())
	}

	for _, class := range rk.GetDefs() {
		fmt.Printf("def: %s\n", class.GetName())
	}

	// TODO: Process records.
}
