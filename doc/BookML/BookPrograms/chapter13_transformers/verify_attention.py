"""Chapter 13: every verification quoted in Sections 13.3 and 13.6."""
import autograd.numpy as np
from autograd import grad
import attention as at

rng = np.random.default_rng(0)
n, d, H = 7, 8, 2
X = rng.normal(size=(n, d))
P = at.init_mha(d, H, rng=rng)
Y, A = at.multihead(P, X)

print("=== Proposition 13.1: attention averages ===")
print(f"rows of A sum to 1 : {np.abs(A.sum(-1) - 1).max():.2e}")
print(f"A >= 0             : {bool((A >= 0).all())}")
V = X @ P["WV"][0]
y0, _ = at.attention(X @ P["WQ"][0], X @ P["WK"][0], V)
inside = ((y0.min(0) >= V.min(0) - 1e-12).all() and
          (y0.max(0) <= V.max(0) + 1e-12).all())
print(f"head output within the bounding box of V : {inside}")

print("\n=== Theorem 13.2: permutation equivariance:  Att(PX) = P Att(X) ===")
r = np.random.default_rng(1)
n, d = 6, 8
X = r.normal(size=(n, d)); P = at.init_mha(d, H, rng=r)
for trial in range(3):
    perm = r.permutation(n); Pm = np.eye(n)[perm]
    Y, _ = at.multihead(P, X); Yp, _ = at.multihead(P, Pm @ X)
    print(f"  trial {trial}: max|Att(PX) - P Att(X)| = "
          f"{np.abs(Yp - Pm @ Y).max():.2e}")

print("\n=== a causal mask breaks it (as it must) ===")
M = at.causal_mask(n)
perm = r.permutation(n); Pm = np.eye(n)[perm]
Y, _ = at.multihead(P, X, M); Yp, _ = at.multihead(P, Pm @ X, M)
print(f"  max|Att(PX) - P Att(X)| with mask = {np.abs(Yp - Pm @ Y).max():.3e}")
print(f"  attention matrix upper triangle (should be 0): "
      f"{np.abs(np.triu(at.multihead(P, X, M)[1][0], 1)).max():.2e}")

print("\n=== Proposition 13.4: variance of the logits ===")
print("  d_k    Var(q.k) unscaled   Var(q.k) scaled   prediction d_k")
for dk in [4, 16, 64, 256, 1024]:
    r2 = np.random.default_rng(2)
    q = r2.normal(size=(4000, dk)); k = r2.normal(size=(4000, dk))
    s = np.sum(q * k, axis=1)
    print(f"  {dk:5d}  {s.var():16.2f}  {(s/np.sqrt(dk)).var():16.4f}  {dk:12d}")

print("\n=== softmax saturation: gradient norm vs logit scale ===")
print("  scale   max A     entropy    ||d softmax/d s||_F")
for c in [0.25, 1.0, 4.0, 16.0, 64.0]:
    r3 = np.random.default_rng(3); s = r3.normal(size=(1, 32)) * c
    a = at.softmax_rows(s)[0]
    J = grad(lambda s: at.softmax_rows(s)[0, 0])(s)
    ent = float(-np.sum(a * np.log(a + 1e-30)))
    print(f"  {c:5.2f}  {a.max():.5f}   {ent:7.4f}   {np.linalg.norm(J):.3e}")

print("\n=== the batched implementation agrees with the looped one ===")
B = 4
Xb = np.random.default_rng(0).normal(size=(B, 6, d))
Pb = at.init_block(d, H, 16, np.random.default_rng(0))
Yb, Ab = at.block_batched(Pb, Xb)
err = 0.0
for b in range(B):
    Y1, A1 = at.block(Pb, Xb[b])
    err = max(err, np.abs(Y1 - Yb[b]).max(), np.abs(A1 - Ab[b]).max())
print(f"  max difference: {err:.2e}")
