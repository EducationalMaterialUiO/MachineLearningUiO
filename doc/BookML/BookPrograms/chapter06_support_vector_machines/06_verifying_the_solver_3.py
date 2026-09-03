"""Chapter 6: listing 6, from the section on verifying the solver.

Extracted from doc/BookML/chapter6.tex.
"""

X, y01 = make_blobs(n_samples=120, centers=2, cluster_std=2.6, random_state=7)
y = np.where(y01 == 0, -1.0, 1.0)

for C in [0.01, 0.1, 1.0, 10.0]:
    m = SVC(C=C, kernel=linear_kernel).fit(X, y)
    w = (m.lam_sv * m.y_sv) @ m.X_sv
    at_bound = int(np.sum(m.lam_ > C - 1e-6))
    print(f"C={C:<6} n_sv={m.sv_.sum():3d} (at bound {at_bound:3d})  "
          f"margin={2/np.linalg.norm(w):7.4f}  "
          f"accuracy={np.mean(m.predict(X) == y):.4f}")
