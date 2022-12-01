
import numpy as np
from scipy import linalg


# from: https://github.com/awesomebytes/parametric_modeling/blob/master/src/by_hand_code/convmtx.py
def py_convmtx(v, n):
    """
    This file is a Python translation of the MATLAB file convm.m
    Python version by RDL 10 Jan 2012
    Copyright notice from convm.m:
    copyright 1996, by M.H. Hayes.  For use with the book
    "Statistical Digital Signal Processing and Modeling"
    (John Wiley & Sons, 1996).
    """
    """
    Generates a convolution matrix

        Usage: X = convm(v,n)
        Given a vector v of length N, an N+n-1 by n convolution matrix is
        generated of the following form:
                  |  v(0)  0      0     ...      0    |
                  |  v(1) v(0)    0     ...      0    |
                  |  v(2) v(1)   v(0)   ...      0    |
             X =  |   .    .      .              .    |
                  |   .    .      .              .    |
                  |   .    .      .              .    |
                  |  v(N) v(N-1) v(N-2) ...  v(N-n+1) |
                  |   0   v(N)   v(N-1) ...  v(N-n+2) |
                  |   .    .      .              .    |
                  |   .    .      .              .    |
                  |   0    0      0     ...    v(N)   |
        And then it's trasposed to fit the MATLAB return value.
        That is, v is assumed to be causal, and zero-valued after N.
    """

    N = len(v) + 2 * n - 2
    xpad = np.concatenate((np.zeros(n - 1), v[:], np.zeros(n - 1)))
    X = np.zeros((len(v) + n - 1, n))

    # Construct X column by column
    for i in range(0, n):
        X[:, i] = xpad[n - i - 1:N - i]

    return X.conj().T


def tls(A, b, thresh=None):
    # Solves the linear equation Ax=b using
    # truncated total least squares.

    m = A.shape[0]
    n = A.shape[1]
    if b.shape != (m, 1):
        print('A, b size mis-match')
        return None

    if thresh is None:
        thresh = np.spacing(1)

    # augmented matrix
    Z = np.concatenate((A.conj().T, b.conj().T), axis=0)  # [full(A.conj().T);b.conj().T];
    U, S, Vh = linalg.svd(Z, lapack_driver='gesvd')
    V = U
    W = S

    # find sing val above
    d = W  # np.array(np.diagonal(W), ndmin=2).conj().T
    k = int(sum(1*(d < thresh)))
    q = n - k + 1

    V12 = V[0:n, (q-1):]  # V(1:n,q:end);
    V22 = V[n:(n+1), (q-1):]  # V(n+1,q:end);

    x = - np.dot(V12, V22.conj().T) / (np.linalg.norm(V22) ** 2)
    return x