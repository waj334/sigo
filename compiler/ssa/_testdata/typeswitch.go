// RUN: FileCheck %s

package main

func typeSwitch() {
	var v any
	switch v.(type) {
	case int:
	case float32:
	case float64:
	default:
	}
}

func typeSwitchAssign(v any) {
	switch v := v.(type) {
	case int:
	case float32, float64:
	case byte:
	default:
		v = v
	}
}

func typeSwitchNoDefault() {
	var v any
	switch v.(type) {
	case int:
	case float32:
	case float64:
	}
}
