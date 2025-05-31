package main

import (
	"regexp"
	"strings"
)

func sanitizeName(display, fallback string) string {
	r := regexp.MustCompile("[\\n\\s]+")
	name := display
	if name == "" {
		name = fallback
	}

	name = r.ReplaceAllString(name, "")
	name = strings.ReplaceAll(name, "%s", "")
	name = strings.ReplaceAll(name, "[%s]", "")
	name = strings.ReplaceAll(name, "[", "")
	name = strings.ReplaceAll(name, "]", "")
	name = strings.ReplaceAll(name, " ", "")
	name = strings.ReplaceAll(name, "-", "_")
	name = strings.ReplaceAll(name, ":", "_")
	name = strings.ReplaceAll(name, ",", "_")
	name = strings.ReplaceAll(name, ".", "_")

	// Clean up trailing separators.
	for strings.HasSuffix(name, "_") {
		name = strings.TrimSuffix(name, "_")
	}

	return name
}

func sanitizeDescription(description string) string {
	r := regexp.MustCompile("[\\n\\s]+")
	// Replace newlines with spaces
	description = r.ReplaceAllString(description, " ")
	description = strings.TrimSpace(description)
	return description
}
