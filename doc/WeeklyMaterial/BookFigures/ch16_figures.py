import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                "..", "BookPrograms", "chapter16_diffusion"))
from common import plt, np, save
import diffusion as D

fig,axs=plt.subplots(1,3,figsize=(13,3.8))
X=np.load("data.npy"); S=np.load("samples_ddpm.npy"); S20=np.load("samples_ddim20.npy")
axs[0].plot(X[:1200,0],X[:1200,1],".",ms=2,color="0.7",label="data")
axs[0].plot(S[:800,0],S[:800,1],".",ms=2,color="C0",label="DDPM, 200 steps")
axs[0].plot(S20[:800,0],S20[:800,1],".",ms=2,color="C3",label="DDIM, 20 steps")
axs[0].set_xlabel("$x_1$"); axs[0].set_ylabel("$x_2$")
axs[0].set_title("(a) samples"); axs[0].legend(fontsize=8,markerscale=4)

steps=[200,50,20,10]; ed=[0.00715,0.01153,0.01096,0.01504]
axs[1].semilogx(steps,ed,"C0o-")
for s,e in zip(steps,ed): axs[1].annotate(f"{s}",(s,e),textcoords="offset points",
                                          xytext=(4,6),fontsize=8)
axs[1].set_xlabel("network calls"); axs[1].set_ylabel("energy distance to the data")
axs[1].set_title("(b) DDIM buys speed cheaply")

T=200
for name,fn,c in [("linear",D.linear_schedule,"C0"),("cosine",D.cosine_schedule,"C1")]:
    b,a,ab=fn(T)
    axs[2].semilogy(np.arange(1,T+1),ab/(1-ab),c,lw=1.6,label=name)
axs[2].axhline(1,color="k",ls=":",lw=0.8)
axs[2].set_xlabel("$t$"); axs[2].set_ylabel(r"$\mathrm{SNR}(t)=\bar\alpha_t/(1-\bar\alpha_t)$")
axs[2].set_title("(c) noise schedules"); axs[2].legend(fontsize=8)
save(fig,16,"diffusion")
