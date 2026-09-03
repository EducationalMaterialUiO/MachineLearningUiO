"""Chapter 9: listing 6, from the section on partial differential equations.

Extracted from doc/BookML/chapter9.tex.
"""

def d_dxk(fun, k):
    """Partial derivative of fun(P, X) with respect to input column k.

    Nesting this gives higher derivatives: d_dxk(d_dxk(u, 0), 0) is
    the second derivative with respect to x_0.
    """
    def wrapped(P, X):
        def scalarised(xk):
            Xn = np.concatenate([X[:, :k], xk.reshape(-1, 1), X[:, k+1:]], axis=1)
            return fun(P, Xn)
        return elementwise_grad(scalarised)(X[:, k])
    return wrapped
