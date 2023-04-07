//sigo:architecture arm
//sigo:cpu cortex-m4
//sigo:triple thumb-none-eabi
package main

import "fmt"

type test struct {
	field0 int
	field1 string
	field2 any
}

func main() {
	i := 0
	fmt.Println("Hello World!!!")
	println(i)
}
