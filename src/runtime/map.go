package runtime

import "unsafe"

const (
	loadFactor = 4
	minBuckets = 8 // Must be of n^2
)

type _map struct {
	state     *mapState
	keyType   *_type
	valueType *_type
}

type mapState struct {
	size     int
	capacity int
	data     []*mapEntry
}

type mapEntry struct {
	hash  uint64
	key   unsafe.Pointer
	value unsafe.Pointer
	next  *mapEntry
}

func mapMake(keyType, valueType *_type, capacity int) _map {
	numBuckets := nextPow2(capacity)
	if numBuckets < minBuckets {
		numBuckets = minBuckets
	}

	initialBuckets := make([]*mapEntry, numBuckets)
	initialCapacity := numBuckets * loadFactor

	m := _map{
		state: &mapState{
			data:     initialBuckets,
			size:     0,
			capacity: initialCapacity,
		},
		keyType:   keyType,
		valueType: valueType,
	}
	return m
}

func mapClear(m _map) {
	if m.state.data == nil {
		return
	}

	numBuckets := nextPow2(m.state.capacity)
	if numBuckets < minBuckets {
		numBuckets = minBuckets
	}

	m.state.data = make([]*mapEntry, numBuckets)
	m.state.size = 0
}

func mapLen(m _map) int {
	return m.state.size
}

func mapUpdate(m _map, key unsafe.Pointer, value unsafe.Pointer) {
	// Perform key lookup
	if entry := _mapLookup(m, key); entry != nil {
		// Update the value
		memcpy(entry.value, value, uintptr(m.valueType.size))
	} else {
		// Resize if necessary
		if m.state.size+1 > m.state.capacity {
			mapResize(m)
		}

		// Calculate hash of the key
		keyHash := mapKeyHash(key, m.keyType)

		// Locate bucket to place value into
		bucketIdx := keyHash % uint64(len(m.state.data))
		bucket := m.state.data[bucketIdx]

		// Insert a new entry into the hash map
		entry = &mapEntry{
			hash:  keyHash,
			key:   alloc(uintptr(m.keyType.size)),
			value: alloc(uintptr(m.valueType.size)),
			next:  bucket,
		}

		// Copy the key and value
		memcpy(entry.key, key, uintptr(m.keyType.size))
		memcpy(entry.value, value, uintptr(m.valueType.size))

		// Update the head of the map
		m.state.data[bucketIdx] = entry

		// Increase the size of the map
		m.state.size++
	}
}

func mapResize(m _map) {
	// double the number of buckets
	oldBuckets := m.state.data
	newBuckets := make([]*mapEntry, 2*len(oldBuckets))

	// rehash the entries
	for _, entry := range oldBuckets {
		for entry != nil {
			hash := entry.hash % uint64(len(newBuckets)) // recompute the hash for the new size
			newBuckets[hash] = &mapEntry{                // prepend the entry to the bucket
				next:  newBuckets[hash],
				key:   entry.key,
				value: entry.value,
			}
			entry = entry.next
		}
	}

	// replace the old bucket slice with the new one
	m.state.data = newBuckets
	m.state.capacity = len(newBuckets) * loadFactor
}

func mapDelete(m _map, key unsafe.Pointer) {
	// Calculate hash of the key
	keyHash := mapKeyHash(key, m.keyType)

	// Locate bucket to place value into
	bucketIdx := keyHash % uint64(len(m.state.data))
	bucket := &m.state.data[bucketIdx]

	// Perform key lookup
	var last *mapEntry
	for entry := *bucket; entry != nil; entry = entry.next {
		if entry.hash == keyHash {
			if entry == *bucket {
				*bucket = entry.next
			} else {
				last.next = entry.next
			}
			return
		}
		last = entry
	}
}

func mapLookup(m _map, key unsafe.Pointer) (unsafe.Pointer, bool) {
	if entry := _mapLookup(m, key); entry != nil {
		return entry.value, true
	}
	return nil, false
}

/*
func mapLookup(m _map, key, result unsafe.Pointer) (unsafe.Pointer, bool) {
	if entry := _mapLookup(m, key); entry != nil {
		// Copy the mapped value to the address given by the result pointer
		memcpy(result, entry.value, uintptr(m.valueType.size))
		return result, true
	}
	// Store the zero value in the result and return false
	memset(result, 0, uintptr(m.valueType.size))
	return result, false
}
*/

func _mapLookup(m _map, K unsafe.Pointer) *mapEntry {
	keyHash := mapKeyHash(K, m.keyType)
	bucketIdx := keyHash % uint64(len(m.state.data))
	bucket := m.state.data[bucketIdx]

	for entry := bucket; entry != nil; entry = entry.next {
		compareResult := false
		if entry.hash == keyHash {
			// Compare the key values just in-case there is a collision with the hash
			switch m.keyType.kind {
			case String:
				// Hash the string's backing array
				lhs := (*_string)(K)
				rhs := (*_string)(entry.key)
				compareResult = stringCompare(*lhs, *rhs)

			//case Interface:
			// TODO: Require the comparison operator for the underlying concrete value type
			case Array:
				arrayType := (*_arrayTypeData)(m.keyType.data)
				// Hash the array's memory as-is
				arraySize := uintptr(arrayType.elementType.size) * uintptr(arrayType.length)
				compareResult = memcmp(entry.key, K, arraySize) == 0
			default:
				compareResult = memcmp(entry.key, K, uintptr(m.keyType.size)) == 0
			}

			if compareResult {
				return entry
			}
		}
	}
	return nil
}

func mapKeyHash(K unsafe.Pointer, T *_type) (result uint64) {
	// Hash the key based on the key's type
	switch T.kind {
	case String:
		// Hash the string's backing array
		str := (*_string)(K)
		result = computeFnv(str.array, uintptr(str.len))
	//case Interface:
	// TODO: Require the comparison operator for the underlying concrete value type
	case Array:
		arrayType := (*_arrayTypeData)(T.data)
		// Hash the array's memory as-is
		result = computeFnv(K, uintptr(arrayType.elementType.size)*uintptr(arrayType.length))
	default:
		result = computeFnv(K, uintptr(T.size))
	}
	return
}

type _mapIterator struct {
	m      _map
	bucket int
	entry  *mapEntry
}

func mapRange(it _mapIterator) (bool, _mapIterator, unsafe.Pointer, unsafe.Pointer) {
	if it.entry == nil {
		// Initialize the iterator by finding the next non-empty bucket
		for ; it.bucket < len(it.m.state.data); it.bucket++ {
			it.entry = it.m.state.data[it.bucket]
			if it.entry != nil {
				break
			}
		}
	}

	if it.entry == nil {
		return false, _mapIterator{}, nil, nil
	} else {
		k := it.entry.key
		v := it.entry.value
		it.entry = it.entry.next
		if it.entry == nil {
			it.bucket++
		}
		return true, it, k, v
	}
}

func mapIsNil(m _map) bool {
	return len(m.state.data) == 0
}

func nextPow2(n int) int {
	n--
	n |= n >> 1
	n |= n >> 2
	n |= n >> 4
	n |= n >> 8
	return n + 1
}
