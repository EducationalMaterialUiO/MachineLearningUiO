"""Diffusion and a normalizing flow on the same two-dimensional target."""
import autograd.numpy as np, time
import diffusion as D, flows
from sklearn.datasets import make_moons

X,_=make_moons(4000,noise=0.06,random_state=0)
X=(X-X.mean(0))/X.std(0)
out=open("cmp.txt","w",buffering=1)

T=200; beta,alpha,abar=D.linear_schedule(T)
P=D.init_eps_net(2,rng=np.random.default_rng(0))
t0=time.time()
P,hd=D.train(P,X,abar,T,n_iter=3000,batch=128,eta=2e-3,rng=np.random.default_rng(1))
out.write(f"diffusion: trained in {time.time()-t0:.1f}s, final L_simple {hd[-1][1]:.4f}\n")

L=flows.init_coupling(2,hidden=64,n_layers=6,rng=np.random.default_rng(0))
t0=time.time()
L,hf=flows.train(L,X,n_iter=3000,batch=128,eta=3e-3,rng=np.random.default_rng(1))
out.write(f"flow: trained in {time.time()-t0:.1f}s, exact log-likelihood {hf[-1][1]:.4f} nats\n\n")

def energy_dist(A,B,n=1500,seed=0):
    """A simple two-sample statistic: lower is closer."""
    r=np.random.default_rng(seed)
    a=A[r.integers(0,len(A),n)]; b=B[r.integers(0,len(B),n)]
    d=lambda U,V: np.mean(np.sqrt(((U[:,None,:]-V[None,:,:])**2).sum(-1)))
    return 2*d(a,b)-d(a,a)-d(b,b)

rng=np.random.default_rng(3)
out.write("  sampler          network calls   energy distance to the data\n")
t0=time.time(); s=D.ddpm_sample(P,1500,2,beta,alpha,abar,T,rng)
out.write(f"  DDPM, T=200      {T:13d}   {energy_dist(X,s):.5f}   ({time.time()-t0:.1f}s)\n")
np.save("samples_ddpm.npy",s)
for k in [50,20,10]:
    t0=time.time(); s=D.ddim_sample(P,1500,2,abar,T,rng,steps=k,eta=0.0)
    out.write(f"  DDIM, {k:3d} steps  {k:13d}   {energy_dist(X,s):.5f}   ({time.time()-t0:.1f}s)\n")
    if k==20: np.save("samples_ddim20.npy",s)
z=np.random.default_rng(5).normal(size=(1500,2)); sf,_=flows.forward(L,z)
out.write(f"  flow             {1:13d}   {energy_dist(X,sf):.5f}\n")
np.save("samples_flow.npy",sf); np.save("data.npy",X)
np.save("hist_d.npy",np.array(hd)); np.save("hist_f.npy",np.array(hf))
out.write("DONE\n")
