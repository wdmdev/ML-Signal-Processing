#%%
import os
import numpy as np
from random import seed
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import signal


seed(42)

# generate signal
nsamp = 500
u = np.random.normal(size = nsamp)

# load impulse response from mat file
irname = 'hpir'
matdata = loadmat(os.path.join(os.path.dirname(__file__), 'data', f'{irname}.mat'))
H = matdata[irname]

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
    e_n = d - w[:,it-1].T @ u_n
    w_n = w_n + (mu / (delta+u_n.T@u_n)) * u_n * e_n

    # store values for later plotting
    e[it-1] = e_n
    w[:, it] = w_n

# plot results
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

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

fig, ax = plt.subplots(2, 1, figsize=(10, 8))
norm_freq, freq_response  = signal.freqz(H)
ax[0].plot(norm_freq, 20 * np.log10(abs(freq_response)), 'b')
ax[0].set_ylabel('Amplitude [dB]', color='b')
ax[0].set_xlabel('Frequency [rad/sample]')
ax[0].grid(axis='both', linestyle='-')

angles = np.unwrap(np.angle(freq_response))
ax[1].plot(norm_freq, angles, 'g')
ax[1].set_ylabel('Angle (radians)', color='g')
ax[1].grid(axis='both', linestyle='-')
ax[1].axis('tight')
plt.show()