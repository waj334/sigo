package runtime

//sigo:linkage rand32 weak
//sigo:export rand32 runtime.rand32
func rand32() uint32 {
	return uint32(rand())
}

//sigo:linkage rand weak
//sigo:export rand runtime.rand
func rand() uint64 {
	g := getg()
	c := &g.chacha8
	for {
		x, ok := c.Next()
		if ok {
			return x
		}
		c.Refill()
	}
}

//sigo:linkage randn weak
//sigo:export randn runtime.randn
func randn(n uint32) uint32 {
	return uint32((uint64(uint32(rand())) * uint64(n)) >> 32)
}
