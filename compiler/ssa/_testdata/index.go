// RUN: FileCheck %s

package main

func indexArray(arr [4]int) int {
	return arr[1]
}

func indexArrayPtr(arr *[4]int) int {
	return arr[2]
}

func indexString(str string) uint8 {
	return str[3]
}

func indexSlice(s []int) int {
	return s[4]
}

func mapIndex(m map[int]int) int {
	if v, ok := m[5]; ok {
		return v
	}
	return m[5]
}

func sliceArray(arr [4]int) ([]int, []int, []int, []int) {
	s0 := arr[0:4]
	s1 := arr[1:]
	s2 := arr[:4]
	s3 := arr[1:2:3]
	return s0, s1, s2, s3
}

func sliceArrayPtr(arr *[4]int) ([]int, []int, []int, []int) {
	s0 := arr[0:4]
	s1 := arr[1:]
	s2 := arr[:4]
	s3 := arr[1:2:3]
	return s0, s1, s2, s3
}

func sliceString(str string) (string, string, string) {
	s0 := str[0:4]
	s1 := str[1:]
	s2 := str[:4]
	return s0, s1, s2
}

func sliceSlice(s []int) ([]int, []int, []int, []int) {
	s0 := s[0:4]
	s1 := s[1:]
	s2 := s[:4]
	s3 := s[1:2:3]
	return s0, s1, s2, s3
}
