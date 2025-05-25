package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	_ "pkg.si-go.dev/sigo/llvm"
	"pkg.si-go.dev/sigo/llvm/tablegen"
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
	// Path of input file to includes.
	currentFile, err := filepath.Abs(input)
	if err != nil {
		log.Fatalln(err)
	}

	includes = append(includes, filepath.Dir(currentFile))

	recordKeeper := tablegen.NewRecordKeeper()
	ok := tablegen.ParseTableGenFile(input, recordKeeper, includes)
	if !ok {
		log.Fatalf("failed to parse the input file\n")
	}

	// Get the series of which code will be generated for.
	// NOTE: Only a single series can be defined!.
	var series *tablegen.Record
	if allSeries := recordKeeper.GetDerivedRecords(constRecordSeries); len(allSeries) == 0 {
		log.Fatalf("no series defined\n")
	} else if len(allSeries) > 1 {
		log.Fatalf("multiple series defined\n")
	} else {
		series = allSeries[0]
	}

	peripheralTypes := series.GetValueAsListOfDefs(constSeriesFieldPeripheralTypes)
	variants := series.GetValueAsListOfDefs(constSeriesFieldVariants)
	arch := series.GetValueAsDef(constSeriesFieldArchitecture)

	peripheralInstances := map[string][]*tablegen.Record{}
	peripheralGroups := map[string][]*tablegen.Record{}

	requiredTags := arch.GetValueAsListOfStrings(constArchitectureTags)
	optionalTags := make([]string, 0, len(variants))
	for _, variant := range variants {
		variantName := variant.GetValueAsString(constObjectFieldName)
		variantName = strings.ToLower(variantName)
		optionalTags = append(optionalTags, variantName)
	}

	// Resolve groups.
	for _, def := range recordKeeper.GetDerivedRecords(constRecordPeripheralGroup) {
		members := def.GetValueAsListOfDefs(constPeripheralGroupFieldInstances)
		if len(members) == 0 {
			continue
		}

		peripheralType := members[0].GetValueAsDef(constPeripheralInstanceFieldType)
		typeRecordName := peripheralType.GetName()

		// Check that all members of the group are the same type.
		for _, member := range members {
			memberType := member.GetValueAsDef(constPeripheralInstanceFieldType)
			memberTypeRecordName := memberType.GetName()
			if memberTypeRecordName != typeRecordName {
				log.Fatalln("mismatch type in peripheral group")
			}
		}

		// Append the peripheral group instance to the slice under the peripheral type's record name.
		s := peripheralGroups[typeRecordName]
		s = append(s, def)
		peripheralGroups[typeRecordName] = s
	}

	for _, def := range recordKeeper.GetDerivedRecords(constRecordPeripheralInstance) {
		peripheralType := def.GetValueAsDef(constPeripheralInstanceFieldType)
		typeRecordName := peripheralType.GetName()

		// Append the peripheral instance to the slice under the peripheral type's record name.
		s := peripheralInstances[typeRecordName]
		s = append(s, def)
		peripheralInstances[typeRecordName] = s
	}

	// Determine which peripherals need to be generated.
	for _, def := range peripheralTypes {
		name := def.GetValueAsString(constObjectFieldName)
		name = strings.ToLower(sanitizeName(name, name))

		filename := filepath.Join(outputDir, "reg", name, "peripheral.go")

		// Create the directory where the peripheral API will be placed.
		err := os.MkdirAll(filepath.Dir(filename), os.ModePerm)
		if err != nil {
			log.Fatalln(err)
		}

		// Create the source file that will be written.
		file, err := os.Create(filename)
		if err != nil {
			log.Fatalln(err)
		}

		instances := peripheralInstances[def.GetName()]
		groups := peripheralGroups[def.GetName()]
		_, err = generatePeripheralType(file, def, instances, groups, requiredTags, optionalTags)
		if err != nil {
			log.Fatalln(err)
		}
	}

	// Generate linker scripts.
	for _, def := range variants {
		name := def.GetValueAsString(constObjectFieldName)
		name = strings.ToLower(sanitizeName(name, name))

		filename := fmt.Sprintf("linker_%s.S", name)
		filename = filepath.Join(outputDir, filename)

		// Create the directory where the peripheral API will be placed.
		err := os.MkdirAll(filepath.Dir(filename), os.ModePerm)
		if err != nil {
			log.Fatalln(err)
		}

		// Create the linker file that will be written.
		file, err := os.Create(filename)
		if err != nil {
			log.Fatalln(err)
		}

		_, err = generateLinkerScript(file, def)
		if err != nil {
			log.Fatalln(err)
		}
	}

	// Generate interrupt vector.
	for _, def := range variants {
		name := def.GetValueAsString(constObjectFieldName)
		name = strings.ToLower(sanitizeName(name, name))

		filename := fmt.Sprintf("isr_%s.S", name)
		filename = filepath.Join(outputDir, filename)

		// Create the directory where the peripheral API will be placed.
		err := os.MkdirAll(filepath.Dir(filename), os.ModePerm)
		if err != nil {
			log.Fatalln(err)
		}

		// Create the assembly file that will be written.
		file, err := os.Create(filename)
		if err != nil {
			log.Fatalln(err)
		}

		_, err = generateArmInterruptVector(file, def)
		if err != nil {
			log.Fatalln(err)
		}
	}

	// Generate init.go.
	{
		filename := filepath.Join(outputDir, "init.go")

		// Create the directory where the peripheral API will be placed.
		err := os.MkdirAll(filepath.Dir(filename), os.ModePerm)
		if err != nil {
			log.Fatalln(err)
		}

		// Create the source file that will be written.
		file, err := os.Create(filename)
		if err != nil {
			log.Fatalln(err)
		}

		_, err = generateSeriesInit(file, series, requiredTags, optionalTags)
		if err != nil {
			log.Fatalln(err)
		}
	}

}
