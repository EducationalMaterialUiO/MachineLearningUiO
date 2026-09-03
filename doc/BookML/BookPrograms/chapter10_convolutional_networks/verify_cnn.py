"""Chapter 10: every verification quoted in Section 10.6.2.

Run with:  python3 verify_cnn.py
Nothing here is asserted in the book that is not printed by this file.
"""
import numpy as np
import cnn

rng = np.random.default_rng(0)


def numgrad(f, A, h=1e-6):
    g = np.zeros_like(A)
    it = np.nditer(A, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        old = A[i]
        A[i] = old + h; fp = f()
        A[i] = old - h; fm = f()
        A[i] = old
        g[i] = (fp - fm) / (2 * h)
        it.iternext()
    return g


# --- 1. forward pass against the definition, Eq. (10.convlayer) --------------
print("=== 1. conv_forward vs the explicit quadruple loop ===")
X = rng.normal(size=(2, 3, 7, 7)); W = rng.normal(size=(4, 3, 3, 3)); b = rng.normal(size=4)
for S, P in [(1, 0), (1, 1), (2, 1), (2, 0), (3, 2)]:
    Y, _ = cnn.conv_forward(X, W, b, S, P)
    N, C, H, Wd = X.shape; K, _, F, _ = W.shape
    H2 = (H - F + 2 * P) // S + 1; W2 = (Wd - F + 2 * P) // S + 1
    Xp = cnn.pad2d(X, P); ref = np.zeros((N, K, H2, W2))
    for n in range(N):
        for k in range(K):
            for a in range(H2):
                for c in range(W2):
                    ref[n, k, a, c] = np.sum(Xp[n, :, a*S:a*S+F, c*S:c*S+F] * W[k]) + b[k]
    print(f"S={S} P={P}  shape {Y.shape}  max|conv-naive| = {np.abs(Y-ref).max():.3e}")

# --- 2. col2im is the adjoint of im2col -------------------------------------
print("\n=== 2. adjoint identity <Ax,y> == <x,A*y> ===")
for S, P, F in [(1, 0, 3), (2, 1, 3), (1, 2, 5)]:
    x = rng.normal(size=(2, 2, 7, 7))
    c = cnn.im2col(x, F, S, P); y = rng.normal(size=c.shape)
    lhs = np.sum(c * y); rhs = np.sum(x * cnn.col2im(y, x.shape, F, S, P))
    print(f"F={F} S={S} P={P}:  |<Ax,y>-<x,A*y>| = {abs(lhs-rhs):.2e}")

# --- 3. gradients against central differences -------------------------------
print("\n=== 3. layer gradients vs finite differences ===")
for S, P in [(1, 0), (1, 1), (2, 1)]:
    X = rng.normal(size=(2, 2, 6, 6)); W = rng.normal(size=(3, 2, 3, 3)); b = rng.normal(size=3)
    Y, cols = cnn.conv_forward(X, W, b, S, P); G = rng.normal(size=Y.shape)
    def loss():
        Yl, _ = cnn.conv_forward(X, W, b, S, P); return float(np.sum(G * Yl))
    dX, dW, db = cnn.conv_backward(G, X, W, cols, S, P)
    print(f"  S={S} P={P}:  dX {np.abs(dX-numgrad(loss,X)).max():.2e}   "
          f"dW {np.abs(dW-numgrad(loss,W)).max():.2e}   "
          f"db {np.abs(db-numgrad(loss,b)).max():.2e}")
X = rng.normal(size=(2, 3, 6, 6))
Y, idx = cnn.maxpool_forward(X, 2, 2); G = rng.normal(size=Y.shape)
def lossp():
    Yl, _ = cnn.maxpool_forward(X, 2, 2); return float(np.sum(G * Yl))
print(f"  max-pool  dX {np.abs(cnn.maxpool_backward(G,X,idx,2,2)-numgrad(lossp,X)).max():.2e}")

# --- 4. structure: Toeplitz and doubly block Toeplitz -----------------------
print("\n=== 4a. polynomial product == convolution == Toeplitz matvec ===")
a = np.array([2., -1., 3.]); bb = np.array([1., 4., -2., 5.])
T = np.zeros((len(a)+len(bb)-1, len(bb)))
for j in range(len(bb)): T[j:j+len(a), j] = a
print("  polymul :", np.polynomial.polynomial.polymul(a, bb))
print("  convolve:", np.convolve(a, bb))
print("  Toeplitz:", T @ bb)
print("  Toeplitz property a_ij = a_{i-j}:",
      all(T[i, j] == T[i+1, j+1] for i in range(T.shape[0]-1) for j in range(T.shape[1]-1)))

print("\n=== 4b. 2-D conv is a doubly block Toeplitz matrix, Eq. (10.unrolled) ===")
X = rng.normal(size=(3, 3)); W = rng.normal(size=(2, 2))
Y, _ = cnn.conv_forward(X[None, None], W[None, None], np.zeros(1), 1, 0)
Wp = np.zeros((4, 9))
for k in range(9):
    e = np.zeros(9); e[k] = 1
    Yk, _ = cnn.conv_forward(e.reshape(1, 1, 3, 3), W[None, None], np.zeros(1), 1, 0)
    Wp[:, k] = Yk.ravel()
print(f"  W'x == vec(Y):  {np.abs(Wp@X.ravel()-Y.ravel()).max()}")
print(f"  {(Wp!=0).sum()} nonzero of {Wp.size}; "
      f"{len(set(np.round(Wp[Wp!=0],10)))} distinct values (the {W.size} kernel entries)")

# --- 5. equivariance, Thm 10.2, and its failure under stride, Prop 10.4 -----
print("\n=== 5. translation equivariance ===")
X = rng.normal(size=(1, 1, 12, 12)); K = rng.normal(size=(1, 1, 3, 3)); b0 = np.zeros(1)
Y, _ = cnn.conv_forward(X, K, b0, 1, 1)
for s in [1, 2, 3]:
    Ys, _ = cnn.conv_forward(np.roll(X, s, axis=3), K, b0, 1, 1)
    sl = (slice(None), slice(None), slice(1, -1), slice(2+s, -2))
    print(f"  S=1 shift {s}: max|conv(T_s x)-T_s conv(x)| = "
          f"{np.abs(Ys[sl]-np.roll(Y,s,axis=3)[sl]).max():.2e}")
Y2, _ = cnn.conv_forward(X, K, b0, 2, 1)
for s in [1, 2]:
    Ys2, _ = cnn.conv_forward(np.roll(X, s, axis=3), K, b0, 2, 1)
    sh = np.roll(Y2, s//2, axis=3)
    tag = "equivariant" if s % 2 == 0 else "NOT equivariant"
    print(f"  S=2 shift {s}: max|.| = "
          f"{np.abs(Ys2[:,:,1:-1,2:-2]-sh[:,:,1:-1,2:-2]).max():.3e}   <- {tag}")
