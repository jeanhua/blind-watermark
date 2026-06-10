package bwm

import "math"

// JacobiSVD computes the SVD of a small square matrix (typically 4x4).
// Returns U, S (diagonal as slice), V^T.
// Uses the Jacobi algorithm for symmetric eigenvalue decomposition of A^T*A.
func JacobiSVD(a [][]float32) (u [][]float32, s []float32, vt [][]float32) {
	n := len(a)
	// Compute A^T * A
	ata := make([][]float32, n)
	for i := 0; i < n; i++ {
		ata[i] = make([]float32, n)
		for j := 0; j < n; j++ {
			var sum float32
			for k := 0; k < n; k++ {
				sum += a[k][i] * a[k][j]
			}
			ata[i][j] = sum
		}
	}
	// Eigenvalue decomposition of ATA using Jacobi
	eigenVals, eigenVecs := jacobiEigen(ata)

	// Singular values = sqrt(eigenvalues), V = eigenvectors
	s = make([]float32, n)
	vt = make([][]float32, n)
	for i := 0; i < n; i++ {
		vt[i] = make([]float32, n)
		if eigenVals[i] > 0 {
			s[i] = float32(math.Sqrt(float64(eigenVals[i])))
		}
		copy(vt[i], eigenVecs[i])
	}

	// U = A * V * S^{-1}
	u = make([][]float32, n)
	for i := 0; i < n; i++ {
		u[i] = make([]float32, n)
		for j := 0; j < n; j++ {
			if s[j] > 1e-10 {
				var sum float32
				for k := 0; k < n; k++ {
					sum += a[i][k] * vt[j][k]
				}
				u[i][j] = sum / s[j]
			}
		}
	}
	return
}

// jacobiEigen computes eigenvalues and eigenvectors of a real symmetric matrix.
// Returns eigenvalues (sorted descending by absolute value) and corresponding eigenvectors as rows of V.
func jacobiEigen(a [][]float32) (eigenVals []float32, eigenVecs [][]float32) {
	n := len(a)
	// Initialize V as identity
	v := make([][]float32, n)
	for i := 0; i < n; i++ {
		v[i] = make([]float32, n)
		v[i][i] = 1
	}
	// Copy A
	mat := make([][]float32, n)
	for i := 0; i < n; i++ {
		mat[i] = make([]float32, n)
		copy(mat[i], a[i])
	}

	const maxIter = 100
	for iter := 0; iter < maxIter; iter++ {
		// Find largest off-diagonal element
		maxVal := float32(0)
		p, q := 0, 1
		for i := 0; i < n; i++ {
			for j := i + 1; j < n; j++ {
				abs := mat[i][j]
				if abs < 0 {
					abs = -abs
				}
				if abs > maxVal {
					maxVal = abs
					p, q = i, j
				}
			}
		}
		if maxVal < 1e-8 {
			break
		}
		// Compute Jacobi rotation
		theta := 0.5 * math.Atan2(float64(2*mat[p][q]), float64(mat[p][p]-mat[q][q]))
		c := float32(math.Cos(theta))
		s := float32(math.Sin(theta))

		// Rotate matrix
		newMat := make([][]float32, n)
		for i := 0; i < n; i++ {
			newMat[i] = make([]float32, n)
			copy(newMat[i], mat[i])
		}
		for i := 0; i < n; i++ {
			if i != p && i != q {
				newMat[i][p] = c*mat[i][p] + s*mat[i][q]
				newMat[p][i] = newMat[i][p]
				newMat[i][q] = -s*mat[i][p] + c*mat[i][q]
				newMat[q][i] = newMat[i][q]
			}
		}
		newMat[p][p] = c*c*mat[p][p] + 2*c*s*mat[p][q] + s*s*mat[q][q]
		newMat[q][q] = s*s*mat[p][p] - 2*c*s*mat[p][q] + c*c*mat[q][q]
		newMat[p][q] = (c*c-s*s)*mat[p][q] + c*s*(mat[q][q]-mat[p][p])
		newMat[q][p] = newMat[p][q]

		mat = newMat

		// Update eigenvectors
		newV := make([][]float32, n)
		for i := 0; i < n; i++ {
			newV[i] = make([]float32, n)
			copy(newV[i], v[i])
		}
		for i := 0; i < n; i++ {
			newV[i][p] = c*v[i][p] + s*v[i][q]
			newV[i][q] = -s*v[i][p] + c*v[i][q]
		}
		v = newV
	}

	// Extract eigenvalues from diagonal
	eigenVals = make([]float32, n)
	for i := 0; i < n; i++ {
		eigenVals[i] = mat[i][i]
	}
	// Transpose V so eigenvectors are rows: V[i] is the i-th eigenvector
	eigenVecs = transpose(v)
	// Sort by eigenvalue magnitude descending
	sortEigen(eigenVals, eigenVecs)
	return
}

func transpose(a [][]float32) [][]float32 {
	n := len(a)
	out := make([][]float32, n)
	for i := 0; i < n; i++ {
		out[i] = make([]float32, n)
		for j := 0; j < n; j++ {
			out[i][j] = a[j][i]
		}
	}
	return out
}

func sortEigen(vals []float32, vecs [][]float32) {
	n := len(vals)
	for i := 0; i < n-1; i++ {
		for j := i + 1; j < n; j++ {
			absI := vals[i]
			if absI < 0 {
				absI = -absI
			}
			absJ := vals[j]
			if absJ < 0 {
				absJ = -absJ
			}
			if absJ > absI {
				vals[i], vals[j] = vals[j], vals[i]
				vecs[i], vecs[j] = vecs[j], vecs[i]
			}
		}
	}
}

// DiagMatrix creates a diagonal matrix from a slice and performs U * diag(s) * V.
func DiagMatMul(u [][]float32, s []float32, v [][]float32) [][]float32 {
	n := len(u)
	// Compute U * diag(S)
	us := make([][]float32, n)
	for i := 0; i < n; i++ {
		us[i] = make([]float32, n)
		for j := 0; j < n; j++ {
			us[i][j] = u[i][j] * s[j]
		}
	}
	// Compute US * V
	out := make([][]float32, n)
	for i := 0; i < n; i++ {
		out[i] = make([]float32, n)
		for j := 0; j < n; j++ {
			var sum float32
			for k := 0; k < n; k++ {
				sum += us[i][k] * v[k][j]
			}
			out[i][j] = sum
		}
	}
	return out
}
