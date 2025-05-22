package main

import (
	"context"
	"flag"
	"log"
	"os"
	"path/filepath"
)

var (
	base            string
	variantPatterns string
	outputDirectory string
)

func init() {
	flag.StringVar(&base, "base", "", "base description file for series")
	flag.StringVar(&variantPatterns, "variants", "", "description file patterns for series variants")
	flag.StringVar(&outputDirectory, "out", "", "output SiGO description file")
	flag.Parse()
}

func main() {
	ctx := context.Background()

	if len(base) == 0 {
		println("no base definition file specified")
		os.Exit(-1)
	}

	var err error
	switch filepath.Ext(base) {
	case ".svd":
		err = translateSVD(ctx, base)
	case ".atdf":
		panic("not implemented")
	default:
		println("unsupported input definition file type")
		os.Exit(-2)
	}

	if err != nil {
		log.Fatalln(err)
	}
}
