#%%
import numpy as np
from random import seed
import matplotlib.pyplot as plt
seed(42)

# generate signal
nsamp = 200
H = np.array([1, 0.8, 0.2])
u = np.random.normal(size = nsamp)

# LMS parms
mu = 0.1
delta = 1e-6

# reserve mem
w = np.zeros((len(H), nsamp+1))
e = np.zeros(nsamp)

# LMS loop
w_n = np.zeros((len(H)))
for it in range(len(H), nsamp+1):
    u_n = np.flip(u[it-len(H):it]).T
    d = H.T @ u_n
    e_n = d - w[:,it-1].T @ u_n # solution 
    w_n = w_n + (mu / (delta+u_n.T@u_n)) * u_n * e_n # solution 

    # store values for later plotting
    e[it-1] = e_n
    w[:, it] = w_n

# plot results
# complete code here
fig, ax = plt.subplots(2, 1, figsize=(10, 8)) # 
ax[0].plot(w.T)
ax[0].set_title('Convergence of parameters')
ax[0].set_xlim((0,nsamp))
ax[0].grid()

ax[1].plot(e)
ax[1].set_title('Residual')
ax[1].set_xlabel('Iteration')
ax[1].set_xlim((0,nsamp))
ax[1].grid()
plt.show() 