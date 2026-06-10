package bwm

import (
	"math/rand/v2"
	"sort"
)

// ShuffleStrategy generates deterministic index permutations for watermark encryption.
// Equivalent to Python: np.random.RandomState(seed).random(size=(numBlocks, blockSize)).argsort(axis=1)
// Each row gives a permutation of indices 0..blockSize-1.
func ShuffleStrategy(seed uint64, numBlocks, blockSize int) [][]int {
	rng := rand.New(rand.NewPCG(seed, seed))
	result := make([][]int, numBlocks)
	for i := 0; i < numBlocks; i++ {
		// Generate random values for argsort
		type pair struct {
			idx int
			val float64
		}
		vals := make([]pair, blockSize)
		for j := 0; j < blockSize; j++ {
			vals[j] = pair{idx: j, val: rng.Float64()}
		}
		sort.Slice(vals, func(a, b int) bool {
			return vals[a].val < vals[b].val
		})
		result[i] = make([]int, blockSize)
		for j := 0; j < blockSize; j++ {
			result[i][j] = vals[j].idx
		}
	}
	return result
}

// ShuffleArray applies a permutation to a slice in-place.
func ShuffleArray(arr []float32, perm []int) []float32 {
	result := make([]float32, len(arr))
	for i, p := range perm {
		result[i] = arr[p]
	}
	return result
}

// UnshuffleArray reverses a permutation on a slice.
func UnshuffleArray(arr []float32, perm []int) []float32 {
	result := make([]float32, len(arr))
	for i, p := range perm {
		result[p] = arr[i]
	}
	return result
}
