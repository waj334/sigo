// RUN: FileCheck %s

package main

func count(yield func(int) bool) {
	for i := 0; i < 3; i++ {
		if !yield(i) {
			return
		}
	}
}

func rangeFuncOne() {
	for v := range count {
		_ = v
	}
}

func pairs(yield func(int, int) bool) {
	for i := 0; i < 3; i++ {
		if !yield(i, i*2) {
			return
		}
	}
}

func rangeFuncTwo() {
	for k, v := range pairs {
		_ = k
		_ = v
	}
}

func ticks(yield func() bool) {
	for i := 0; i < 3; i++ {
		if !yield() {
			return
		}
	}
}

func rangeFuncZero() {
	count := 0
	for range ticks {
		count++
	}
	_ = count
}
