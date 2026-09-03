# `universal.py` — the quantitative side of the universal approximation theorem

This program supplies the numbers quoted in Section 8.4 of Chapter 8.  It
checks four statements that the classical theorems of Cybenko and Hornik do
not make, all of which are proved in the chapter:

1. **Theorem 8.relurate, the constructive width.**  A single hidden layer of
   `N` rectified units interpolating an `L`-Lipschitz `F` at `N+1` equispaced
   knots has supremum error at most `L(b-a)/2N`.  The program *writes the
   weights down* — nothing is trained — and measures the error, for a smooth
   target where the bound is loose and for the worst-case triangular wave
   where it is attained exactly (ratio `1.0000`).

2. **Representability is not learnability.**  The same architecture at the
   same widths, fitted by Adam from a random start, with PyTorch's default
   initialisation and with the kinks spread uniformly over the interval as the
   construction places them.  Training never reaches the constructed network
   and falls further behind as the width grows.

3. **Theorem 8.depth, the depth separation.**  The sawtooth
   `s_k = T ∘ ... ∘ T` with `T(x) = 2 ReLU(x) − 4 ReLU(x−1/2)` is computed
   exactly by a network of depth `k` with two units per layer, while any
   single hidden layer needs at least `2^k − 1` units to come within `1/2`.
   The program counts the linear pieces of both and finds the wall.

4. **Theorem 8.barron.**  For a target whose Barron norm `C_F` is known in
   closed form, the mean squared error is measured in `d = 1`, `5` and `20`
   and compared with the bound `(2 r C_F)^2 / N`.  The rate does not change
   with the dimension.

## Running it

```
python3 universal.py           # writes universal.txt and the .npy files
```

Needs `numpy` and `torch`; it runs on the CPU in about twenty minutes and is
fully deterministic (every seed is fixed), so the numbers reproduce exactly.
Everything it prints to `universal.txt` is quoted verbatim in the chapter.

Afterwards,

```
cd ../../BookFigures && python3 ch08_universal_figures.py
```

builds `chapter08_neural_networks/universal_construction.pdf` and
`depth_separation.pdf` from the saved arrays.
