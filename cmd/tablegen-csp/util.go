package main

import (
	"errors"
	"strings"
)

func formatGoIdentifier(identifier string, exported bool) string {
	if exported {
		return strings.ToUpper(identifier[:1]) + identifier[1:]
	} else {
		return strings.ToLower(identifier[:1]) + identifier[1:]
	}
}

func formatCamelCase(s ...string) string {
	var builder strings.Builder
	for i, term := range s {
		term = strings.ToLower(term)
		if i > 0 {
			// Upper case the following terms.
			term = strings.ToUpper(term[:1]) + term[1:]
		}
		builder.WriteString(term)
	}
	return builder.String()
}

func typeForWidth(width int64) (string, error) {
	switch {
	case width <= 8:
		return "uint8", nil
	case width <= 16:
		return "uint16", nil
	case width <= 32:
		return "uint32", nil
	case width <= 64:
		return "uint64", nil
	default:
		return "", errors.New("invalid width")
	}
}
