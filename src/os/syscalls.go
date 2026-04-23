package os

import "unsafe"

// POSIX I/O syscalls required by picolibc's tinystdio POSIX bridge.
// These back printf/fprintf and other stdio functions on bare-metal targets.

//go:export goWrite write
func goWrite(fd int32, buf unsafe.Pointer, count int32) int32 {
	if (fd == 1 || fd == 2) && Stdout != nil {
		b := unsafe.Slice((*byte)(buf), count)
		n, _ := Stdout.Write(b)
		return int32(n)
	}
	return count // discard
}

//go:export goRead read
func goRead(fd int32, buf unsafe.Pointer, count int32) int32 {
	if fd == 0 && Stdin != nil {
		b := unsafe.Slice((*byte)(buf), count)
		n, _ := Stdin.Read(b)
		return int32(n)
	}
	return -1
}

//go:export goLseek lseek
func goLseek(fd int32, offset int32, whence int32) int32 {
	return -1
}

//go:export goClose close
func goClose(fd int32) int32 {
	return 0
}
