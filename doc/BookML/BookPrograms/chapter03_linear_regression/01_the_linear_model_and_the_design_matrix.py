"""Chapter 3: listing 1, from the section on the linear model and the design matrix.

Extracted from doc/BookML/chapter3.tex.
"""

import numpy as np

def design_matrix_2d(x, y, degree):
    """Design matrix for a two-dimensional polynomial of the given degree.

    Columns are the monomials x^a y^b with a + b <= degree, ordered by
    total degree.  The first column is the intercept.
    """
    x, y = np.ravel(x), np.ravel(y)
    columns = []
    for total in range(degree + 1):
        for b in range(total + 1):
            columns.append(x**(total - b) * y**b)
    return np.column_stack(columns)


# A degree-5 fit in two variables has (5+1)(5+2)/2 = 21 parameters
X = design_matrix_2d(np.random.rand(100), np.random.rand(100), degree=5)
print(X.shape)
