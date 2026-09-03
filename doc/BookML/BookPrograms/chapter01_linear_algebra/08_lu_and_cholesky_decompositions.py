"""Chapter 1: listing 8, from the section on lu and cholesky decompositions.

Extracted from doc/BookML/chapter1.tex.
"""

class LUDecomposition(DirectSolver):
    """Doolittle LU factorisation with partial pivoting, P A = L U."""

    def _factorize(self):
        n = self.n
        LU = self.A.copy()
        perm = np.arange(n)
        sign = 1.0

        for k in range(n):
            p = k + np.argmax(np.abs(LU[k:, k]))
            if abs(LU[p, k]) < 1.0e-14:
                raise np.linalg.LinAlgError("matrix is singular")
            if p != k:
                LU[[k, p]] = LU[[p, k]]
                perm[[k, p]] = perm[[p, k]]
                sign = -sign
            # the multipliers l_ik are stored in place, below the diagonal
            LU[k+1:, k] /= LU[k, k]
            # rank-one update of the trailing submatrix
            LU[k+1:, k+1:] -= np.outer(LU[k+1:, k], LU[k, k+1:])

        self.LU, self.perm, self.sign = LU, perm, sign

    def solve(self, b):
        """Solve A x = b in two steps: L y = P b, then U x = y."""
        y = np.asarray(b, dtype=float)[self.perm].copy()
        n = self.n
        for i in range(1, n):                    # forward, L has unit diagonal
            y[i] -= self.LU[i, :i] @ y[:i]
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):           # backward
            x[i] = (y[i] - self.LU[i, i+1:] @ x[i+1:]) / self.LU[i, i]
        return x

    def determinant(self):
        return self.sign * np.prod(np.diag(self.LU))
