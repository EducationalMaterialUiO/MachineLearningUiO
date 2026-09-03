import autograd.numpy as np, time
import attention as at, recall
out=open("recall_res.txt","w",buffering=1)
out.write("  L    T   model         params   test accuracy (2 seeds)\n")
res={}
for L in [2,4,8]:
    T=2*L+1
    d=32; PE=at.positional_encoding(T,d)
    accs=[]
    for seed in range(2):
        P=recall.init_transformer(d,2,64,np.random.default_rng(seed))
        npar=recall.n_params(P)
        t0=time.time()
        P,a=recall.train(P,recall.transformer_logits,L,d,n_iter=3000,batch=64,
                         eta=3e-3,seed=seed,PE=PE)
        accs.append(a)
    res[("tr",L)]=accs
    out.write(f"  {L}   {T:2d}   transformer   {npar:6d}   "
              f"{np.mean(accs):.3f}  ({min(accs):.3f}-{max(accs):.3f})\n")
    accs=[]
    for seed in range(2):
        Q=recall.init_rnn(86,np.random.default_rng(seed))
        npar2=recall.n_params(Q)
        Q,a=recall.train(Q,recall.rnn_logits,L,86,n_iter=3000,batch=64,
                         eta=3e-3,seed=seed)
        accs.append(a)
    res[("rnn",L)]=accs
    out.write(f"  {L}   {T:2d}   RNN           {npar2:6d}   "
              f"{np.mean(accs):.3f}  ({min(accs):.3f}-{max(accs):.3f})\n")
out.write(f"chance = {1/recall.NSYM:.3f}\nDONE\n")
import json; json.dump({f"{k[0]}{k[1]}":v for k,v in res.items()},open("recall.json","w"))
