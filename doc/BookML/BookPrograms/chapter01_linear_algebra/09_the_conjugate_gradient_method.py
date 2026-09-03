"""Chapter 1: listing 9, from the section on the conjugate gradient method.

Extracted from doc/BookML/chapter1.tex.
"""

def conjugate_gradient(A, b, x0=None, tol=1.0e-10, maxiter=None):
    """Solve A x = b for symmetric positive definite A.

    A may be a matrix or any callable implementing the product A @ v,
    which is what makes the method usable when A is never formed.
    """
    matvec = A if callable(A) else (lambda v: A @ v)
    n = b.shape[0]
    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).copy()

    r = b - matvec(x)
    p = r.copy()
    rsold = r @ r

    for _ in range(maxiter or n):
        Ap = matvec(p)
        alpha = rsold / (p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        rsnew = r @ r
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p        # Fletcher-Reeves update
        rsold = rsnew

    return x
