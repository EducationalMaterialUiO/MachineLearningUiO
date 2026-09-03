"""Translated digits: the experiment that separates a CNN from a dense net."""
import numpy as np, cnn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def place(imgs, rng, out=12):
    """Put each 8x8 digit at a random offset inside an out x out canvas."""
    n=len(imgs); C=np.zeros((n,1,out,out))
    for k in range(n):
        i,j=rng.integers(0,out-8+1,size=2)
        C[k,0,i:i+8,j:j+8]=imgs[k]
    return C

d=load_digits(); imgs=d.images/16.0; y=d.target
itr,ite,ytr,yte=train_test_split(imgs,y,test_size=0.3,random_state=42,stratify=y)
Ytr=np.eye(10)[ytr]

def dense(Xtr,Ytr,Xte,yte,hidden,seed,epochs=20):
    rng=np.random.default_rng(seed)
    Xf=Xtr.reshape(len(Xtr),-1); Xef=Xte.reshape(len(Xte),-1); D=Xf.shape[1]
    P={"W1":rng.normal(0,np.sqrt(2/D),(D,hidden)),"b1":np.zeros(hidden),
       "W2":rng.normal(0,np.sqrt(2/hidden),(hidden,10)),"b2":np.zeros(10)}
    m={k:np.zeros_like(v) for k,v in P.items()}; v_={k:np.zeros_like(v) for k,v in P.items()}; t=0
    for ep in range(epochs):
        order=rng.permutation(len(Xf))
        for s in range(0,len(order),32):
            i=order[s:s+32]; xb,yb=Xf[i],Ytr[i]; n=len(i)
            Z1=xb@P["W1"]+P["b1"]; A1=cnn.relu(Z1); pr=cnn.softmax(A1@P["W2"]+P["b2"])
            d2=(pr-yb)/n; g={"W2":A1.T@d2,"b2":d2.sum(0)}
            d1=(d2@P["W2"].T)*cnn.relu_prime(Z1); g["W1"]=xb.T@d1; g["b1"]=d1.sum(0)
            t+=1
            for k in P:
                m[k]=0.9*m[k]+0.1*g[k]; v_[k]=0.999*v_[k]+0.001*g[k]**2
                P[k]-=3e-3*(m[k]/(1-0.9**t))/(np.sqrt(v_[k]/(1-0.999**t))+1e-8)
    pr=cnn.softmax(cnn.relu(Xef@P["W1"]+P["b1"])@P["W2"]+P["b2"])
    return float((pr.argmax(1)==yte).mean()), sum(v.size for v in P.values())

print("canvas  model   params   mean acc over 5 seeds (min-max)")
for out,hid,flat in [(8,25,64),(12,44,144)]:
    ca,da=[],[]
    for seed in range(5):
        rng=np.random.default_rng(100+seed)
        Xtr=place(itr,rng,out) if out>8 else itr[:,None]
        Xte=place(ite,rng,out) if out>8 else ite[:,None]
        p=cnn.init_cnn(flat=16*(out//4)**2,rng=np.random.default_rng(seed))
        p,h=cnn.train(p,Xtr,Ytr,Xte,yte,epochs=20,batch=32,eta=3e-3,
                      rng=np.random.default_rng(seed),verbose=False)
        ca.append(h[-1][2]); npar_c=cnn.n_params(p)
        a,npar_d=dense(Xtr,Ytr,Xte,yte,hid,seed); da.append(a)
    ca,da=np.array(ca),np.array(da)
    print(f"{out}x{out}   CNN     {npar_c:6d}   {ca.mean():.4f}  ({ca.min():.4f}-{ca.max():.4f})")
    print(f"{out}x{out}   dense   {npar_d:6d}   {da.mean():.4f}  ({da.min():.4f}-{da.max():.4f})")
