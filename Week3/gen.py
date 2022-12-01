import numpy as np
from scipy import signal

# Generate gaussian white noise, Mean=0 Var=1
def gen(N, a):
    noise = np.random.normal(size=N)
    x = signal.lfilter([1], [1, -a], noise)
    return x