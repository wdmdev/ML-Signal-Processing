import numpy as np
def mse(xhat, x):
    # MSE Compute the (optimally scaled) mean squared error
    # between a (noisy) signal estimate and an original (noise free) signal
    #          _              _
    #         |             2  |
    # e = min | (x - a*xhat)   |
    #      a  |_              _|
    #
    # Usage
    #    e = mse(xhat, x);
    #
    # Inputs
    #    xhat    Estimated signal
    #    x       Original signal
    # 
    # Outputs
    #    e       Mean squared error
    #
    # Copyright 2013 Mikkel N. Schmidt, mnsc@dtu.dk
    # 2022, translated to Python by Tommy S. Alstrøm, tsal@dtu.dk

    a = (x.T @ xhat) / (xhat.T @ xhat)
    e = np.mean((x-a*xhat)**2)
    return e