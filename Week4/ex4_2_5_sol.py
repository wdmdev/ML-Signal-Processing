#%%
import numpy as np
from random import seed
from scipy import signal
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
from xcorr import xcorr
seed(42)

nsamp = 200 # solution 
H = np.array([1, 0.8, 0.2]) # solution 
u = np.random.normal(size = nsamp)
d = signal.lfilter(H, [1], u)
l = 3
r_uu = xcorr(u,u,l-1) # solution 
r_du = xcorr(d,u,l-1) # solution 
r_uu_inx = toeplitz(np.arange(l-1,l*2-1))
R_uu = r_uu[r_uu_inx]
w = np.linalg.solve(R_uu, r_du[l-1:2*l-1])
err = np.mean((H-w)**2)

print("The filter parameters are: ", w)
print(f"Error of weights: {err:0.3}")