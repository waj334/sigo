package main

import (
	"golang.org/x/text/cases"
	"golang.org/x/text/language"
	"io"
	"slices"
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

func writeBuildTags(output io.StringWriter, device device.Device, mandatoryTags []string) (int, error) {
	var builder strings.Builder
	var variantTags []string

	if len(device.Series) > 0 {
		mandatoryTags = append([]string{device.Series}, mandatoryTags...)
	}

	for _, tag := range device.BuildTags {
		if slices.Contains(mandatoryTags, tag) {
			continue
		}
		mandatoryTags = append(mandatoryTags, tag)
	}

	for _, variant := range device.Variants {
		variantTags = append(variantTags, variant.Identifier)
	}

	if len(mandatoryTags) > 0 || len(variantTags) > 0 {
		builder.WriteString("//go:build ")

		if len(mandatoryTags) > 0 {
			for i, tag := range mandatoryTags {
				builder.WriteString(tag)
				if i != len(mandatoryTags)-1 {
					builder.WriteString(" && ")
				}
			}

			if len(variantTags) > 0 {
				builder.WriteString(" && ")
			}
		}

		if len(variantTags) > 0 {
			builder.WriteString("(")
			for i, tag := range variantTags {
				builder.WriteString(tag)
				if i != len(variantTags)-1 {
					builder.WriteString(" || ")
				}
			}
			builder.WriteString(")")
		}

		builder.WriteString("\n\n")
	}

	return output.WriteString(builder.String())
}
