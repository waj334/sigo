package runtime

// Map Runtime Implementation
//
// This implementation provides pointer-based map access semantics:
//
// - mapAddr(m, key) -> *value
//   Returns a pointer to the value slot for the given key.
//   If the key doesn't exist, allocates a new entry with zero value.
//   Used by compiler for both reads (v := m[k]) and writes (m[k] = v).
//
// - mapLookup(m, key) -> (*value, bool)
//   Returns pointer to value and true if key exists, nil and false otherwise.
//   Does NOT allocate if key is missing.
//   Used by compiler for comma-ok idiom (v, ok := m[k]).
//
// The compiler generates:
//   m[k] = v    →  *mapAddr(m, &k) = v
//   v := m[k]   →  v := *mapAddr(m, &k)
//   v, ok := m[k] → vPtr, ok := mapLookup(m, &k); v := *vPtr (if ok)

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

// mapAddr returns a pointer to the value slot for the given key.
// If the key doesn't exist, it allocates a new entry with a zero value.
// The compiler uses this for both reads (m[k]) and writes (m[k] = v).
//
//go:export mapAddr runtime.mapAddr
func mapAddr(m _map, key unsafe.Pointer) unsafe.Pointer {
	// Perform key lookup
	if entry := _mapLookup(m, key); entry != nil {
		// Key exists, return pointer to value slot
		return entry.value
	}

	// Key doesn't exist - allocate a new entry
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
	entry := &mapEntry{
		hash:  keyHash,
		key:   alloc(uintptr(m.keyType.size)),
		value: alloc(uintptr(m.valueType.size)),
		next:  bucket,
	}

	// Copy the key (value starts as zero from alloc)
	memcpy(entry.key, key, uintptr(m.keyType.size))

	// Update the head of the bucket
	m.state.data[bucketIdx] = entry

	// Increase the size of the map
	m.state.size++

	// Return pointer to the value slot
	return entry.value
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

// mapLookup performs a lookup with the comma-ok idiom: v, ok := m[k]
// Returns a pointer to the value and true if the key exists, or nil and false if not.
// Unlike mapAddr, this does NOT allocate a new entry if the key is missing.
//
//go:export mapLookup runtime.mapLookup
func mapLookup(m _map, key unsafe.Pointer) (unsafe.Pointer, bool) {
	if entry := _mapLookup(m, key); entry != nil {
		// Return the pointer to the value slot (not a copy of the value)
		return entry.value, true
	}
	return nil, false
}

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

func mapRangeInit(m _map) _mapIterator {
	var it _mapIterator
	it.m = m

	// Initialize the iterator by finding the next non-empty bucket
	mapRangeNext(&it)
	return it
}

func mapRangeNext(it *_mapIterator) {
	if it.m.state == nil {
		return
	}

	for ; it.bucket < len(it.m.state.data); it.bucket++ {
		it.entry = it.m.state.data[it.bucket]
		if it.entry != nil {
			break
		}
	}
}

func mapRange(it *_mapIterator) (unsafe.Pointer, unsafe.Pointer, bool) {
	if it.entry == nil {
		return nil, nil, false
	} else {
		k := it.entry.key
		v := it.entry.value
		it.entry = it.entry.next
		if it.entry == nil {
			it.bucket++

			// Find the next non-empty bucket.
			mapRangeNext(it)
		}
		return k, v, true
	}
}

func mapIsNil(m _map) bool {
	return m.state == nil
}

func nextPow2(n int) int {
	n--
	n |= n >> 1
	n |= n >> 2
	n |= n >> 4
	n |= n >> 8
	return n + 1
}
