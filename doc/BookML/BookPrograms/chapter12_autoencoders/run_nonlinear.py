"""A one-dimensional curved manifold in three dimensions: PCA cannot follow it."""
import numpy as np, ae

def spiral(n, noise=0.03, seed=0):
    r = np.random.default_rng(seed)
    t = r.uniform(0, 1, n)
    X = np.column_stack([np.cos(3*np.pi*t), np.sin(3*np.pi*t), t*2.0])
    return X + noise*r.normal(size=X.shape), t

X, t = spiral(800); Xte, tte = spiral(400, seed=7)
mu = X.mean(0); X = X-mu; Xte = Xte-mu
out = open("nl_res.txt","w",buffering=1)
out.write(f"data: {X.shape[0]} points on a 1-D curve in R^3, test {Xte.shape[0]}\n")
var = np.mean(np.sum(X**2,axis=1))
for p in [1,2]:
    Pi,Up,lam = ae.pca(X,p)
    e_tr = ae.mse(X@Pi,X); e_te = ae.mse(Xte@Pi,Xte)
    out.write(f"PCA p={p}: train {e_tr:.5f}  test {e_te:.5f}  "
              f"({100*e_te/var:.1f}% of total variance unexplained)\n")
acts = ["tanh","tanh","tanh","identity"]
for p in [1,2]:
    errs=[]
    for seed in range(3):
        P = ae.init_ae([3,16,p,16,3], acts, np.random.default_rng(seed))
        P,h = ae.train_ae(P,X,acts,n_epoch=600,batch=32,eta=5e-3,
                          rng=np.random.default_rng(seed),Xval=Xte)
        errs.append((ae.mse(ae.ae_forward(P,X,acts)[0],X),
                     ae.mse(ae.ae_forward(P,Xte,acts)[0],Xte)))
        if seed==0 and p==1:
            np.save("nl_hist.npy",np.array(h)); np.save("nl_X.npy",X); np.save("nl_t.npy",t)
            np.save("nl_rec.npy",ae.ae_forward(P,X,acts)[0])
            np.save("nl_code.npy",ae.encode(P,X,acts,2))
            np.save("nl_pca1.npy",X@ae.pca(X,1)[0])
    a=np.array(errs)
    out.write(f"AE  p={p}: train {a[:,0].mean():.5f}  test {a[:,1].mean():.5f}  "
              f"(3 seeds, test range {a[:,1].min():.5f}-{a[:,1].max():.5f})\n")
out.write(f"total variance per point: {var:.5f}\n")
out.write("DONE\n")
