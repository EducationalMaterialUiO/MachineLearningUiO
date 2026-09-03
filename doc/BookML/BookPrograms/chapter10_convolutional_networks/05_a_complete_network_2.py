"""Chapter 10: listing 5, from the section on a complete network.

Extracted from doc/BookML/chapter10.tex.
"""

def init_cnn(C_in=1, K1=8, K2=16, F=3, n_out=10, flat=None, rng=None):
    """He initialisation, Eq. (8.he), with fan-in C*F*F for a conv kernel."""
    rng = np.random.default_rng(0) if rng is None else rng
    def he(shape, fan_in):
        return rng.normal(0.0, np.sqrt(2.0 / fan_in), shape)
    return {
        "W1": he((K1, C_in, F, F), C_in * F * F), "b1": np.zeros(K1),
        "W2": he((K2, K1, F, F), K1 * F * F),     "b2": np.zeros(K2),
        "W3": he((flat, n_out), flat),            "b3": np.zeros(n_out),
    }
