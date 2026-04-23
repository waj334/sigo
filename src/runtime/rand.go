package runtime

var runtimeRandState uint64 = 0x123456789ABCDEF0

//sigo:export runtimeRand runtime.rand
func runtimeRand() uint64 {
	s := runtimeRandState
	s ^= s << 13
	s ^= s >> 7
	s ^= s << 17
	runtimeRandState = s
	return s
}
