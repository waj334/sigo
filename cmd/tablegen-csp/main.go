package main

import (
	"flag"
	"fmt"
	"log"
	"omibyte.io/sigo/llvm/tablegen"
	"os"
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
	recordKeeper := tablegen.NewRecordKeeper()
	ok := tablegen.ParseTableGenFile(input, recordKeeper, includes)
	if !ok {
		log.Fatalf("failed to parse the input file")
	}

	peripheralInstances := map[string][]tablegen.Record{}
	peripheralGroups := map[string][]tablegen.Record{}

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
				log.Fatal("mismatch type in peripheral group")
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
	for _, def := range recordKeeper.GetDerivedRecords(constRecordPeripheralType) {
		instances := peripheralInstances[def.GetName()]
		groups := peripheralGroups[def.GetName()]
		_, err := generatePeripheralType(os.Stdout, def, instances, groups)
		if err != nil {
			log.Fatal(err)
		}
	}
}
