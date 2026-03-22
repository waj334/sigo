package runtime

import "unsafe"

//sigo:extern abort runtime.abort
func abort()

//sigo:extern exec runtime.exec
func exec(args, fn unsafe.Pointer)

//sigo:export exit _exit
func exit(code int) {
	abort()
}
