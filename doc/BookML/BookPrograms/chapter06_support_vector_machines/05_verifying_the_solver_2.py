"""Chapter 6: listing 5, from the section on verifying the solver.

Extracted from doc/BookML/chapter6.tex.
"""

X, y01 = make_moons(n_samples=200, noise=0.15, random_state=42)
y = np.where(y01 == 0, -1.0, 1.0)

gaussian = lambda A, B: rbf_kernel(A, B, gamma=1.0)
model = SVC(C=1.0, kernel=gaussian).fit(X, y)

margin = y * model.decision_function(X)                 # y_i f(x_i)
lam = model.lam_
free = (lam > 1e-6) & (lam < model.C - 1e-6)

print(f"free support vectors: |y f - 1| max = "
      f"{np.abs(margin[free] - 1).max():.2e}")          # must be 0
print(f"lambda = 0 points:    min y f = {margin[lam <= 1e-8].min():.4f}")  # >= 1
print(f"equality constraint:  sum lam y = {np.sum(lam * y):.2e}")          # = 0

dual = np.sum(lam) - 0.5 * np.sum(np.outer(lam * y, lam * y) * model.K_)
print(f"dual objective: {dual:.6f}")
