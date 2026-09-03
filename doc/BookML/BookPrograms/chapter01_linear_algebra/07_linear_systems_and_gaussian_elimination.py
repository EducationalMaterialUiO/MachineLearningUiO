"""Chapter 1: listing 7, from the section on linear systems and gaussian elimination.

Extracted from doc/BookML/chapter1.tex.
"""

class GaussianElimination(DirectSolver):
    """Gaussian elimination with partial pivoting."""

    def solve(self, b):
        n = self.n
        M = self.A.copy()                 # the elimination destroys its input
        y = np.asarray(b, dtype=float).copy()

        for k in range(n - 1):            # forward elimination
            # partial pivoting: use the largest element in the column
            p = k + np.argmax(np.abs(M[k:, k]))
            if abs(M[p, k]) < 1.0e-14:
                raise np.linalg.LinAlgError("matrix is singular")
            if p != k:
                M[[k, p]] = M[[p, k]]
                y[k], y[p] = y[p], y[k]
            for i in range(k + 1, n):
                factor = M[i, k] / M[k, k]
                M[i, k:] -= factor * M[k, k:]
                y[i] -= factor * y[k]

        return self._back_substitute(M, y)

    @staticmethod
    def _back_substitute(U, y):
        """Solve U x = y for an upper triangular U."""
        n = U.shape[0]
        x = np.zeros(n)
        for m in range(n - 1, -1, -1):
            x[m] = (y[m] - U[m, m+1:] @ x[m+1:]) / U[m, m]
        return x
