package runtime

import "unsafe"

const (
	loadFactor = 4
	minBuckets = 8 // Must be of n^2
)

type _map struct {
	buckets   *[]*mapEntry
	size      *int
	capacity  *int
	keyType   *_type
	valueType *_type
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
	initialSize := 0

	return _map{
		buckets:   &initialBuckets,
		size:      &initialSize,
		capacity:  &initialCapacity,
		keyType:   keyType,
		valueType: valueType,
	}
}

func mapLen(m _map) int {
	return *m.size
}

func mapUpdate(m _map, key unsafe.Pointer, value unsafe.Pointer) {
	// Perform key lookup
	if entry := _mapLookup(m, key); entry != nil {
		// Update the value
		memcpy(entry.value, value, m.valueType.size)
	} else {
		// Resize if necessary
		if *m.size+1 > *m.capacity {
			mapResize(&m)
		}

		// Calculate hash of the key
		keyHash := mapKeyHash(key, m.keyType)

		// Locate bucket to place value into
		bucketIdx := keyHash % uint64(len(*m.buckets))
		bucket := &(*m.buckets)[bucketIdx]

		// Insert a new entry into the hash map
		entry := &mapEntry{
			hash:  keyHash,
			key:   alloc(m.keyType.size),
			value: alloc(m.valueType.size),
		}

		if *bucket != nil {
			entry.next = (*bucket).next
		}

		// Copy the key and value
		memcpy(entry.key, key, m.keyType.size)
		memcpy(entry.value, value, m.valueType.size)

		// Update the head of the map
		*bucket = entry

		// Increase the size of the map
		*m.size++
	}
}

func mapResize(m *_map) {
	// double the number of buckets
	oldBuckets := *m.buckets
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
	*m.buckets = newBuckets
	*m.capacity = len(newBuckets) * loadFactor
}

func mapDelete(m _map, key unsafe.Pointer) {
	// Calculate hash of the key
	keyHash := mapKeyHash(key, m.keyType)

	// Locate bucket to place value into
	bucketIdx := keyHash % uint64(len(*m.buckets))
	bucket := &(*m.buckets)[bucketIdx]

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

func mapLookup(m _map, key unsafe.Pointer, result unsafe.Pointer) (ok bool) {
	if entry := _mapLookup(m, key); entry != nil {
		// Copy the mapped value to the address given by the result pointer
		memcpy(result, entry.value, m.valueType.size)
		return true
	}
	// Store the zero value in the result and return false
	memset(result, 0, m.valueType.size)
	return false
}

func _mapLookup(m _map, K unsafe.Pointer) *mapEntry {
	keyHash := mapKeyHash(K, m.keyType)
	bucketIdx := keyHash % uint64(len(*m.buckets))
	bucket := (*m.buckets)[bucketIdx]

	for entry := bucket; entry != nil; entry = entry.next {
		compareResult := false
		if entry.hash == keyHash {
			// Compare the key values just in-case there is a collision with the hash
			switch m.keyType.construct {
			case Primitive:
				switch m.keyType.kind {
				case String:
					// Hash the string's backing array
					lhs := (*_string)(K)
					rhs := (*_string)(entry.key)
					compareResult = *lhs == *rhs
				default:
					compareResult = memcmp(entry.key, K, m.keyType.size) == 0
				}
			//case Interface:
			// TODO: Require the comparison operator for the underlying concrete value type
			case Array:
				// Hash the array's memory as-is
				arraySize := m.keyType.array.elementType.size * uintptr(m.keyType.array.capacity)
				compareResult = memcmp(entry.key, K, arraySize) == 0
			default:
				compareResult = memcmp(entry.key, K, m.keyType.size) == 0
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
	switch T.construct {
	case Primitive:
		switch T.kind {
		case String:
			// Hash the string's backing array
			str := (*_string)(K)
			result = computeFnv(str.array, uintptr(str.len))
		default:
			result = computeFnv(K, T.size)
		}
	//case Interface:
	// TODO: Require the comparison operator for the underlying concrete value type
	case Array:
		// Hash the array's memory as-is
		result = computeFnv(K, T.array.elementType.size*uintptr(T.array.capacity))
	default:
		result = computeFnv(K, T.size)
	}
	return
}

func nextPow2(n int) int {
	return n + (2 - (n % 2))
}
