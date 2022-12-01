import numpy as np
from convmtx import convmtx
# Input
#     x: input signal
#     y: desired signal
#     L: filter length
#     mu: step size
#
# Output
#     yhat: filter output
#
# 2020-2022 Tommy Sonne Alstrøm, tsal@dtu.dk

def nlms(x, y, L, mu, delta):
    N = y.shape[0]
    w = np.zeros(L,)
    yhat = np.zeros(N,)
    e = np.zeros(N,)

    # if x is a one-dimensional vector we assume x is a time series and we
    # create X accordingly using X = convmtx(x, L), otherwise we assume it
    # already is a convolution matrix
    if x.ndim == 1:
        X = convmtx(x, L).T
    else:
        X = x

    for n in range(0, N):
        x_n = X[:,n]
        yhat[n] = w.T @ x_n
        e[n] = y[n] - yhat[n]
        w = w + (mu / (delta+x_n.T@x_n)) * x_n * e[n]
        
    return yhat, e