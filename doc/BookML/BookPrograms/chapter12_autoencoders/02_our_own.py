"""Chapter 12: listing 2, from the section on our own.

Extracted from doc/BookML/chapter12.tex.
"""

acts = ["tanh", "tanh", "tanh", "identity"]
P = init_ae([3, 16, 1, 16, 3], acts, np.random.default_rng(0))
P, hist = train_ae(P, X, acts, n_epoch=600, batch=32, eta=5e-3, Xval=Xte)
Z = encode(P, X, acts, layer=2)          # the code: run the first two layers
