package main

import (
	"golang.org/x/text/cases"
	"golang.org/x/text/language"
	"io"
	"strings"

	"omibyte.io/sigo/targets/device"
)

func formatSymbol(input string, exported bool) string {
	var builder strings.Builder
	caser := cases.Title(language.English)

	// Replace all underscores with spaces.
	input = strings.ReplaceAll(input, "_", " ")

	// Split the input string into a slice of fields substrings.
	words := strings.Fields(input)
	for i, word := range words {
		if i == 0 && !exported {
			word = strings.ToLower(word)
		} else {
			word = caser.String(word)
		}
		builder.WriteString(word)
	}

	result := builder.String()
	return result
}

func writeBuildTags(output io.StringWriter, device device.Device) (int, error) {
	var builder strings.Builder
	var tags []string

	if len(device.Series) > 0 {
		tags = append(tags, device.Series)
	}

	for _, variant := range device.Variants {
		tags = append(tags, variant.Identifier)
	}

	if len(device.Series) > 0 || len(tags) > 0 {
		builder.WriteString("//go:build ")
	}

	if len(device.Series) > 0 {
		builder.WriteString(device.Series)
		if len(tags) > 0 {
			builder.WriteString(" && ")
		}
	}

	if len(tags) > 0 {
		builder.WriteString("(")
		for i, tag := range tags {
			builder.WriteString(tag)
			if i != len(tags)-1 {
				builder.WriteString(" || ")
			}
		}
		builder.WriteString(")\n\n")
	}

	return output.WriteString(builder.String())
}
