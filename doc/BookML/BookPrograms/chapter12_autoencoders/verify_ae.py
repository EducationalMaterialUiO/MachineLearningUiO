"""Chapter 12: the verifications quoted in Sections 12.5 and 12.7."""
import numpy as np
import ae

# ---------------------------------------------------------------------------
# 1. gradients
# ---------------------------------------------------------------------------
def gradcheck(sizes, acts, seed=1, n=40, d=None, tol=1e-8):
    rng = np.random.default_rng(0)
    d = sizes[0]
    X = rng.normal(size=(n, d)); X -= X.mean(0)
    P = ae.init_ae(sizes, acts, np.random.default_rng(seed))
    Xh, c = ae.ae_forward(P, X, acts)
    g = ae.ae_backward(P, c, Xh, X, acts)

    def loss():
        return ae.cost(ae.ae_forward(P, X, acts)[0], X)

    worst_rel, n_skipped, worst_abs = 0.0, 0, 0.0
    for l in range(len(P)):
        for j in range(2):
            A = P[l][j]
            it = np.nditer(A, flags=["multi_index"])
            while not it.finished:
                i = it.multi_index; old = A[i]; h = 1e-6
                A[i] = old + h; fp = loss()
                A[i] = old - h; fm = loss()
                A[i] = old
                num = (fp - fm) / (2 * h); ana = g[l][j][i]
                worst_abs = max(worst_abs, abs(num - ana))
                if max(abs(num), abs(ana)) < tol:      # both zero: see the notebox
                    n_skipped += 1
                else:
                    worst_rel = max(worst_rel, abs(num - ana) /
                                    (abs(num) + abs(ana)))
                it.iternext()
    return worst_rel, worst_abs, n_skipped


# ---------------------------------------------------------------------------
# 2. the linear autoencoder against PCA
# ---------------------------------------------------------------------------
def principal_angles(A, B):
    """Principal angles in degrees between the column spaces of A and B."""
    QA, _ = np.linalg.qr(A); QB, _ = np.linalg.qr(B)
    s = np.linalg.svd(QA.T @ QB, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1, 1)))


def linear_ae_vs_pca(d=8, p=3, n=600, seed=0, n_epoch=800, eta=5e-3):
    rng = np.random.default_rng(seed)
    # data with a genuine p-dimensional structure plus isotropic noise
    B = rng.normal(size=(p, d))
    Z = rng.normal(size=(n, p)) * np.array([4.0, 2.0, 1.0])[:p]
    X = Z @ B + 0.25 * rng.normal(size=(n, d))
    X -= X.mean(0)

    Pi, Up, lam = ae.pca(X, p)
    acts = ["identity", "identity"]
    P = ae.init_ae([d, p, d], acts, np.random.default_rng(seed + 1))
    P, hist = ae.train_ae(P, X, acts, n_epoch=n_epoch, batch=32, eta=eta,
                          rng=np.random.default_rng(seed + 2))
    Xhat_ae = ae.ae_forward(P, X, acts)[0]
    Xhat_pca = X @ Pi
    We, Wd = P[0][0], P[1][0]           # (d,p) and (p,d)
    return dict(X=X, lam=lam, Up=Up, Pi=Pi, We=We, Wd=Wd, P=P, hist=hist,
                Xhat_ae=Xhat_ae, Xhat_pca=Xhat_pca, p=p, d=d)


if __name__ == "__main__":
    print("=== 1. gradient check, all parameters ===")
    for sizes, acts in [([6, 3, 6], ["identity", "identity"]),
                        ([6, 4, 2, 4, 6], ["tanh", "tanh", "tanh", "sigmoid"]),
                        ([6, 3, 6], ["relu", "identity"])]:
        r, a, k = gradcheck(sizes, acts)
        print(f"  {str(sizes):18s} {str(acts):46s}")
        print(f"      max relative error {r:.2e}   max absolute error {a:.2e}")
        if k:
            print(f"      ({k} entries had gradient 0 and were checked absolutely)")

    print("\n=== 2. a linear autoencoder recovers the PCA subspace, Thm 12.4 ===")
    R = linear_ae_vs_pca(n_epoch=3000, eta=1e-2)
    lam, p = R["lam"], R["p"]
    ae_err = ae.mse(R["Xhat_ae"], R["X"])
    pca_err = ae.mse(R["Xhat_pca"], R["X"])
    print(f"AE  reconstruction error : {ae_err:.6f}")
    print(f"PCA reconstruction error : {pca_err:.6f}")
    print(f"eigenvalue tail          : {lam[p:].sum():.6f}")
    print(f"AE/PCA error ratio       : {ae_err/pca_err:.6f}")
    ang = principal_angles(R["Wd"].T, R["Up"])
    print(f"principal angles (deg)   : {np.array2string(ang, precision=4)}")
    A = R["We"] @ R["Wd"]
    print(f"||W_e W_d - Pi||_F       : {np.linalg.norm(A - R['Pi']):.3e}")
    sv = np.linalg.svd(A, compute_uv=False)
    print(f"singular values of W_e W_d: {np.array2string(sv, precision=4)}")
    print(f"||W_e - U_p||_F          : {np.linalg.norm(R['We']-R['Up']):.4f}"
          f"  (NOT small: Prop 12.3)")
    res = np.linalg.norm(R["We"] - R["Up"] @ (np.linalg.pinv(R["Up"]) @ R["We"]))
    print(f"dist(W_e, span U_p)      : {res:.3e}  (small: same subspace)")
