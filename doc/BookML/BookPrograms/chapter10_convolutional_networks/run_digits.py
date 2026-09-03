import numpy as np, time, cnn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
d=load_digits(); X=(d.images/16.0)[:,None,:,:]; y=d.target
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
Ytr=np.eye(10)[ytr]

print("=== CNN: conv(1->8) pool conv(8->16) pool dense(64->10) ===")
accs=[]
for seed in range(5):
    p=cnn.init_cnn(flat=64,rng=np.random.default_rng(seed))
    t0=time.time()
    p,hist=cnn.train(p,Xtr,Ytr,Xte,yte,epochs=20,batch=32,eta=3e-3,
                     rng=np.random.default_rng(seed),verbose=(seed==0))
    accs.append(hist[-1][2])
    if seed==0:
        np.save("hist_cnn.npy",np.array(hist)); print(f"  params {cnn.n_params(p)}  time {time.time()-t0:.1f}s")
        np.savez("params_cnn.npz",**p)
accs=np.array(accs)
print(f"CNN over 5 seeds: mean {accs.mean():.4f}  min {accs.min():.4f}  max {accs.max():.4f}")

# --- dense baseline with matched parameter count ---
print()
print("=== dense baseline, matched parameters ===")
def dense_run(hidden,seed):
    rng=np.random.default_rng(seed)
    Xf=Xtr.reshape(len(Xtr),-1); Xef=Xte.reshape(len(Xte),-1)
    W1=rng.normal(0,np.sqrt(2/64),(64,hidden)); b1=np.zeros(hidden)
    W2=rng.normal(0,np.sqrt(2/hidden),(hidden,10)); b2=np.zeros(10)
    P={"W1":W1,"b1":b1,"W2":W2,"b2":b2}
    m={k:np.zeros_like(v) for k,v in P.items()}; v_={k:np.zeros_like(v) for k,v in P.items()}
    t=0; hist=[]
    for ep in range(20):
        order=rng.permutation(len(Xf))
        for s in range(0,len(order),32):
            i=order[s:s+32]; xb,yb=Xf[i],Ytr[i]; n=len(i)
            Z1=xb@P["W1"]+P["b1"]; A1=cnn.relu(Z1)
            pr=cnn.softmax(A1@P["W2"]+P["b2"])
            d2=(pr-yb)/n
            g={"W2":A1.T@d2,"b2":d2.sum(0)}
            d1=(d2@P["W2"].T)*cnn.relu_prime(Z1)
            g["W1"]=xb.T@d1; g["b1"]=d1.sum(0)
            t+=1
            for k in P:
                m[k]=0.9*m[k]+0.1*g[k]; v_[k]=0.999*v_[k]+0.001*g[k]**2
                P[k]-=3e-3*(m[k]/(1-0.9**t))/(np.sqrt(v_[k]/(1-0.999**t))+1e-8)
        Z1=Xef@P["W1"]+P["b1"]; pr=cnn.softmax(cnn.relu(Z1)@P["W2"]+P["b2"])
        hist.append((ep+1,0.0,float((pr.argmax(1)==yte).mean())))
    return hist, sum(v.size for v in P.values())
for hidden in [25]:
    a=[]
    for seed in range(5):
        h,np_=dense_run(hidden,seed); a.append(h[-1][2])
        if seed==0: np.save("hist_dense.npy",np.array(h))
    a=np.array(a)
    print(f"dense hidden={hidden} params {np_}: mean {a.mean():.4f}  min {a.min():.4f}  max {a.max():.4f}")
