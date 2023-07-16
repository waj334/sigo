package atomic

//sigo:linkname __atomic_compare_exchange_4 __atomic_compare_exchange_4
//sigo:linkname __atomic_load_4 __atomic_load_4
//sigo:linkname __atomic_store_4 __atomic_store_4
//sigo:linkname __atomic_fetch_add_4 __atomic_fetch_add_4
//sigo:linkname __atomic_compare_exchange_8 __atomic_compare_exchange_8
//sigo:linkname __atomic_load_8 __atomic_load_8
//sigo:linkname __atomic_store_8 __atomic_store_8
//sigo:linkname __atomic_fetch_add_8 __atomic_fetch_add_8

//sigo:linkage __atomic_compare_exchange_4 weak
//sigo:linkage __atomic_load_4 weak
//sigo:linkage __atomic_store_4 weak
//sigo:linkage __atomic_fetch_add_4 weak
//sigo:linkage __atomic_compare_exchange_8 weak
//sigo:linkage __atomic_load_8 weak
//sigo:linkage __atomic_store_8 weak
//sigo:linkage __atomic_fetch_add_8 weak

//sigo:extern enableInterrupts _enable_irq
//sigo:extern disableInterrupts _disable_irq

func enableInterrupts(state uint32)
func disableInterrupts() uint32

func __atomic_compare_exchange_4(ptr *uint32, expected *uint32, desired uint32, successOrder int, failureOrder int) bool {
	state := disableInterrupts()
	current := *ptr
	if current == *expected {
		*ptr = desired
		enableInterrupts(state)
		return true
	} else {
		*expected = current
		enableInterrupts(state)
		return false
	}
}

func __atomic_compare_exchange_8(ptr *uint64, expected *uint64, desired uint64, successOrder int, failureOrder int) bool {
	state := disableInterrupts()
	current := *ptr
	if current == *expected {
		*ptr = desired
		enableInterrupts(state)
		return true
	} else {
		*expected = current
		enableInterrupts(state)
		return false
	}
}

func __atomic_load_4(ptr *uint32, ordering int) uint32 {
	state := disableInterrupts()
	value := *ptr
	enableInterrupts(state)
	return value
}

func __atomic_load_8(ptr *uint64, ordering int) uint64 {
	state := disableInterrupts()
	value := *ptr
	enableInterrupts(state)
	return value
}

func __atomic_store_4(ptr *uint32, val uint32, ordering int) {
	state := disableInterrupts()
	*ptr = val
	enableInterrupts(state)
}

func __atomic_store_8(ptr *uint64, val uint64, ordering int) {
	state := disableInterrupts()
	*ptr = val
	enableInterrupts(state)
}

func __atomic_fetch_add_4(ptr *uint32, val uint32, ordering int) uint32 {
	state := disableInterrupts()
	value := *ptr
	*ptr = value + val
	enableInterrupts(state)
	return value
}

func __atomic_fetch_add_8(ptr *uint64, val uint64, ordering int) uint64 {
	state := disableInterrupts()
	value := *ptr
	*ptr = value + val
	enableInterrupts(state)
	return value
}
