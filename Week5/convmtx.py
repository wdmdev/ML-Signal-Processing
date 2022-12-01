import numpy as np
import scipy.linalg as linalg

def convmtx(h, n):
    '''
    Convolution matrix, same as convmtx does in matlab
    '''
    return linalg.toeplitz(
        np.hstack([h, np.zeros(n-1)]),
        np.hstack([h[0], np.zeros(n-1)]),
    )