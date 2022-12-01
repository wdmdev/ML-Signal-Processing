import numpy as np
from convmtx import convmtx

def rls(x, y, L, beta, lambda_):
    '''
    Input
        x: input signal
        y: desired signal
        L: filter length
        beta: forget factor
        lambda_: regularization

    Output
        yhat: filter output
    '''
    # reserve mem
    yhat = np.zeros(len(y))
    e = np.zeros(len(y))

    # if x is a one-dimensional vector we assume x is a time series and we
    # create X accordingly using X = convmtx(x, L), otherwise we assume it
    # already is a convolution matrix
    if x.ndim == 1:
        X = np.fliplr(convmtx(x,L))
    else:
        X = x

    # start RLS
    # initialize
    w = np.zeros(L)  # theta in the book
    w = np.expand_dims(w, -1)
    P = 1/lambda_*np.eye(L)

    # for each n do
    for n in range(len(y)):
        # get x_n
        x_n = X[n, :]
        x_n = np.expand_dims(x_n, -1)

        # get filter output
        yhat[n] = w.T@x_n

        # update iteration
        e[n] = y[n] - yhat[n]
        denum = beta + x_n.T@P@x_n
        K_n = (P@x_n)/denum
        w = w + K_n*e[n]
        P = (P - (K_n @ x_n.T) @ P)/beta

    return yhat, e