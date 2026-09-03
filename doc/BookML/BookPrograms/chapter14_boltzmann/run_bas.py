"""Bars and stripes on a 3x3 grid: small enough that log Z is exact."""
import numpy as np, rbm

def bars_and_stripes(L=3):
    """All images that are constant along rows, or constant along columns."""
    pats=set()
    for m in range(2**L):
        bits=[(m>>i)&1 for i in range(L)]
        pats.add(tuple(np.repeat(bits,L)))              # bars  (rows constant)
        pats.add(tuple(np.tile(bits,L)))                # stripes (cols constant)
    return np.array(sorted(pats),dtype=float)

D=bars_and_stripes(3); M=D.shape[1]
X=D[np.random.default_rng(0).integers(0,len(D),400)]

def train(mode,k,n_iter=3000,eta=0.05,seed=0,N=8):
    rng=np.random.default_rng(seed)
    P=rbm.init_rbm(M,N,np.random.default_rng(seed),scale=0.1)
    hist=[]
    for it in range(1,n_iter+1):
        idx=rng.integers(0,len(X),32); xb=X[idx]
        g = rbm.exact_gradient(P,xb) if mode=="exact" else rbm.cd_gradient(P,xb,k=k,rng=rng)
        for kk in P: P[kk]+=eta*g[kk]
        if it%100==0: hist.append((it,rbm.log_likelihood(P,X)))
    return P,hist

out=open("bas_res.txt","w",buffering=1)
out.write(f"bars and stripes 3x3: {len(D)} distinct patterns of {2**M} possible\n")
out.write(f"log-likelihood of the perfect model: {np.log(1/len(D)):.4f} per image\n\n")
out.write("  training signal    final log-likelihood (3 seeds)\n")
res={}
for name,mode,k in [("CD-1","cd",1),("CD-10","cd",10),("exact gradient","exact",0)]:
    lls=[]
    for seed in range(3):
        P,h=train(mode,k,seed=seed); lls.append(h[-1][1])
        if seed==0: res[name]=h
    a=np.array(lls)
    out.write(f"  {name:16s}   {a.mean():.4f}   ({a.min():.4f} to {a.max():.4f})\n")
np.save("bas_hist.npy",np.array([[h[1] for h in res[n]] for n in res]))
np.save("bas_it.npy",np.array([h[0] for h in res["CD-1"]]))
np.save("bas_D.npy",D)
out.write("DONE\n")
