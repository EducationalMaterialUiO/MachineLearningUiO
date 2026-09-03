"""Chapter 6: listing 7, from the section on verifying the solver.

Extracted from doc/BookML/chapter6.tex.
"""

rng = np.random.default_rng(1)
a, b = rng.normal(size=2), rng.normal(size=2)
phi = lambda v: np.array([v[0]**2, v[1]**2, np.sqrt(2) * v[0] * v[1]])

print(phi(a) @ phi(b), (a @ b)**2)      # 0.9148995105  0.9148995105
