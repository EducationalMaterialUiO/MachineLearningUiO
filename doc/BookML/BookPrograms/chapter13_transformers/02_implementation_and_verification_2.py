"""Chapter 13: listing 2, from the section on implementation and verification.

Extracted from doc/BookML/chapter13.tex.
"""

def block(P, X, mask=None):
    """Pre-norm transformer block, Eq. (13.block).

    X -> X + MHA(LN(X)) -> X + MLP(LN(X)).  The residual paths carry the
    identity, which is what keeps deep stacks trainable (cf. Section 11.lstmwhy).
    """
    Y, A = multihead(P, layernorm(X, P["g1"], P["be1"]), mask)
    X = X + Y
    Z = layernorm(X, P["g2"], P["be2"])
    X = X + gelu(Z @ P["W1"] + P["b1"]) @ P["W2"] + P["b2"]
    return X, A
