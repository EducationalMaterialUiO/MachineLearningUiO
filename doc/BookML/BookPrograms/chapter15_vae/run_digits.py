"""A VAE on binarised 8x8 digits, and a measurement of posterior collapse."""
import autograd.numpy as np
import vae
from sklearn.datasets import load_digits

d=load_digits()
X=(d.images.reshape(len(d.images),-1)/16.0)
X=(X>0.5).astype(float)                       # binarise: Bernoulli decoder
ntr=1200; Xtr,Xte=X[:ntr],X[ntr:]
out=open("digits_res.txt","w",buffering=1)
out.write(f"binarised 8x8 digits: {Xtr.shape[0]} train, {Xte.shape[0]} test, d={X.shape[1]}\n\n")
out.write("  d_h   test ELBO   active units (KL_j > 0.01)   mean KL per unit\n")
res={}
for dh in [2,5,10,20]:
    P=vae.init_vae(X.shape[1],dh,hidden=64,rng=np.random.default_rng(0))
    P,h=vae.train_vae(P,Xtr,n_iter=4000,batch=64,eta=2e-3,
                      rng=np.random.default_rng(1),every=200)
    e=np.random.default_rng(7).normal(size=(len(Xte),dh))
    te=float(vae.elbo(P,Xte,e))
    mu,lv=vae.encode(P,Xte)
    klj=0.5*np.mean(mu**2+np.exp(lv)-lv-1.0,axis=0)     # KL per latent unit
    active=int(np.sum(klj>0.01))
    out.write(f"  {dh:3d}   {te:9.4f}   {active:23d}   {klj.mean():.4f}\n")
    res[dh]=(np.array(h),klj)
    if dh==10:
        np.save("vae_hist.npy",np.array(h)); np.save("vae_klj.npy",klj)
        np.save("vae_mu.npy",mu); np.save("vae_y.npy",d.target[ntr:])
np.save("vae_klall.npy",np.array([np.pad(res[k][1],(0,20-len(res[k][1]))) for k in res]))
out.write("DONE\n")
