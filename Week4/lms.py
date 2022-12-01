import numpy as np
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

def lms(x, y, L, mu):
    N = np.size(x, 0)
    w = np.zeros(L,)
    yhat = np.zeros(N,)    

    # zero-pad input signal
    x = np.concatenate((np.zeros(L-1,), x), axis=0)

    for n in range(0, N):
        x_n = x[n:n+L]
        yhat[n] = w.T @ x_n
        e = y[n] - yhat[n]
        w = w + mu*e*x_n
        
    return yhat