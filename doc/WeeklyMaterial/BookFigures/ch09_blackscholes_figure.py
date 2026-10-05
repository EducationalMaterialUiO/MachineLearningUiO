import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                "..", "BookPrograms", "chapter09_differential_equations"))
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import plt, np, save
import pickle, bs
from scipy.stats import norm
from nn_de import network, d_dxk
P=pickle.load(open("bs_P.pkl","rb"))
pred=np.load("bs_pred.npy"); ex=np.load("bs_exact.npy")
n=120; Sv=np.linspace(0,bs.S_max,n); tv=np.linspace(0,bs.T,n)
Pm=pred.reshape(n,n); Em=ex.reshape(n,n)

fig,axs=plt.subplots(1,3,figsize=(13,3.8))
for t,col in [(0.0,"C0"),(0.5,"C1"),(0.95,"C2")]:
    j=int(t/bs.T*(n-1))
    axs[0].plot(Sv,Pm[:,j],col,lw=1.7,label=f"$t={t}$")
    axs[0].plot(Sv,Em[:,j],col,ls="--",lw=1.0)
axs[0].plot(Sv,np.maximum(Sv-bs.K,0),"k:",lw=1.0,label="payoff")
axs[0].axvline(bs.K,color="gray",lw=0.7)
axs[0].set_xlabel("$S$"); axs[0].set_ylabel("$C(S,t)$")
axs[0].set_title("(a) network (solid) vs Black-Scholes (dashed)")
axs[0].legend(fontsize=8)

c=axs[1].contourf(Sv,tv,np.abs(Pm-Em).T,levels=20,cmap="cividis")
plt.colorbar(c,ax=axs[1]); axs[1].grid(False)
axs[1].plot([bs.K],[bs.T],"r*",ms=13)
axs[1].annotate("kink of the payoff\n$S=K$, $t=T$",xy=(bs.K,bs.T),xytext=(9.5,0.62),
                color="white",fontsize=8.5,
                arrowprops=dict(arrowstyle="->",color="white",lw=1.0))
axs[1].set_xlabel("$S$"); axs[1].set_ylabel("$t$")
axs[1].set_title("(b) absolute error")

Cn=lambda P,X: network(P,X,"tanh"); dC=d_dxk(Cn,0)
Ss=np.linspace(0.05,bs.S_max,200)
X=np.column_stack([Ss/bs.K, bs.T*np.ones_like(Ss)])
d1=(np.log(Ss/bs.K)+(bs.r+0.5*bs.sigma**2)*bs.T)/(bs.sigma*np.sqrt(bs.T))
axs[2].plot(Ss,dC(P,X),"C0",lw=1.7,label=r"$\partial N/\partial S$, network")
axs[2].plot(Ss,norm.cdf(d1),"k--",lw=1.2,label=r"$\Delta=N(d_1)$, exact")
axs[2].axhline(0,color="gray",lw=0.6); axs[2].axvline(bs.K,color="gray",lw=0.7)
axs[2].set_xlabel("$S$"); axs[2].set_ylabel(r"$\Delta$")
axs[2].set_title(r"(c) the Greek $\Delta$, free by differentiation")
axs[2].legend(fontsize=8,loc="lower right")
save(fig,9,"blackscholes")
