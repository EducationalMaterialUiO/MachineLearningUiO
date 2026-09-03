"""Chapter 6: listing 3, from the section on implementation.

Extracted from doc/BookML/chapter6.tex.
"""

class SVC:
    """Support vector classifier trained by sequential minimal optimisation.

    Solves the dual (6.dualkernel) subject to the box constraints
    (6.boxconstraints), using the analytic two-variable update (6.smoupdate).
    Labels must be -1 and +1.
    """

    def __init__(self, C=1.0, kernel=linear_kernel, tol=1e-4, max_iter=10000):
        self.C = C
        self.kernel = kernel
        self.tol = tol
        self.max_iter = max_iter

    # ---------- the two-variable subproblem ----------
    def _take_step(self, i, j):
        if i == j:
            return False
        C, lam, y, K, E = self.C, self.lam_, self.y_, self.K_, self.E_
        li, lj = lam[i], lam[j]

        # Bounds L and H, Eqs. (6.boundsdiff) and (6.boundssame)
        if y[i] != y[j]:
            L, H = max(0.0, lj - li), min(C, C + lj - li)
        else:
            L, H = max(0.0, li + lj - C), min(C, li + lj)
        if L >= H:
            return False

        eta = 2.0 * K[i, j] - K[i, i] - K[j, j]        # Eq. (6.eta)
        if eta >= -1e-12:                              # degenerate, skip
            return False

        lj_new = lj - y[j] * (E[i] - E[j]) / eta       # Eq. (6.smoupdate)
        lj_new = min(H, max(L, lj_new))                # Eq. (6.clip)
        if abs(lj_new - lj) < 1e-10 * (lj_new + lj + 1e-10):
            return False
        li_new = li + y[i] * y[j] * (lj - lj_new)      # Eq. (6.smopartner)

        b1 = (self.b_ - E[i] - y[i] * (li_new - li) * K[i, i]
              - y[j] * (lj_new - lj) * K[i, j])        # Eq. (6.b1)
        b2 = (self.b_ - E[j] - y[i] * (li_new - li) * K[i, j]
              - y[j] * (lj_new - lj) * K[j, j])        # Eq. (6.b2)
        if 1e-8 < li_new < C - 1e-8:
            b_new = b1
        elif 1e-8 < lj_new < C - 1e-8:
            b_new = b2
        else:
            b_new = 0.5 * (b1 + b2)

        lam[i], lam[j] = li_new, lj_new
        self.b_ = b_new
        self.E_ = (self.K_ @ (lam * y)) + self.b_ - y   # refresh the cache
        return True

    # ---------- choose the partner for a violating index ----------
    def _examine(self, i):
        y, lam, C = self.y_, self.lam_, self.C
        r = y[i] * self.E_[i]
        violates = ((r < -self.tol and lam[i] < C - 1e-12) or
                    (r > self.tol and lam[i] > 1e-12))   # Eq. (6.kktviolation)
        if not violates:
            return False

        free = np.where((lam > 1e-12) & (lam < C - 1e-12))[0]
        if len(free) > 1:                       # heuristic: maximise |E_i - E_j|
            j = free[np.argmax(np.abs(self.E_[i] - self.E_[free]))]
            if self._take_step(i, j):
                return True
        for j in np.random.permutation(free):   # then any free multiplier
            if self._take_step(i, j):
                return True
        for j in np.random.permutation(len(y)):  # then anything at all
            if self._take_step(i, j):
                return True
        return False

    # ---------- the outer loop ----------
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)

        self.X_, self.y_ = X, y
        self.K_ = self.kernel(X, X)
        self.lam_ = np.zeros(n)
        self.b_ = 0.0
        self.E_ = -y.copy()                    # since lambda = 0 and b = 0

        examine_all, num_changed, it = True, 0, 0
        while (num_changed > 0 or examine_all) and it < self.max_iter:
            num_changed = 0
            if examine_all:
                index_set = range(n)
            else:                              # only the free support vectors
                index_set = np.where((self.lam_ > 1e-12)
                                     & (self.lam_ < self.C - 1e-12))[0]
            for i in index_set:
                num_changed += self._examine(i)
                it += 1
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

        sv = self.lam_ > 1e-8
        self.sv_ = sv
        self.X_sv, self.y_sv, self.lam_sv = X[sv], y[sv], self.lam_[sv]
        self.n_iter_ = it
        return self

    def decision_function(self, X):
        """Eq. (6.kernelclassify) without the sign."""
        X = np.asarray(X, dtype=float)
        return self.kernel(X, self.X_sv) @ (self.lam_sv * self.y_sv) + self.b_

    def predict(self, X):
        return np.sign(self.decision_function(X))
