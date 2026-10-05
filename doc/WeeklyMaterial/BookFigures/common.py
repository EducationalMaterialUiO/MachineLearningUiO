import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, os
plt.rcParams.update({"figure.dpi":110,"savefig.dpi":200,"font.size":10,
    "axes.grid":True,"grid.alpha":0.3,"axes.spines.top":False,"axes.spines.right":False,
    "figure.autolayout":True,"legend.frameon":False})
BASE=os.path.join(os.path.dirname(os.path.abspath(__file__)))
DIRS={1:"chapter01_linear_algebra",2:"chapter02_statistics",3:"chapter03_linear_regression",
      4:"chapter04_optimization",5:"chapter05_logistic_regression",
      6:"chapter06_support_vector_machines",7:"chapter07_trees_and_ensembles",8:"chapter08_neural_networks",9:"chapter09_differential_equations",10:"chapter10_convolutional_networks",11:"chapter11_recurrent_networks",12:"chapter12_autoencoders",13:"chapter13_transformers",14:"chapter14_boltzmann",15:"chapter15_vae",16:"chapter16_diffusion",17:"chapter17_gan"}
def save(fig, ch, name):
    d=f"{BASE}/{DIRS[ch]}"; os.makedirs(d,exist_ok=True)
    fig.savefig(f"{d}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{d}/{name}.png", bbox_inches="tight")
    plt.close(fig); print(f"  {DIRS[ch]}/{name}.pdf")
