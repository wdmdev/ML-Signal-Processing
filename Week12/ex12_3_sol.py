#%%
import os

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# data parameters
np.random.seed(0)
N = 100
samples = 1000
percent_outlier = 0.1
snr = 10  # dB

# learning parameters
sigma = 0.004
C = 1e-2

# init figure
fig, ax = plt.subplots()

# load bladerunner data
indices = np.arange(0, samples, samples//N)
start = 100000
sound, fs = sf.read(os.path.join(DATA_DIR, 'BladeRunner.wav'), frames=samples, start=start)
y = sound[indices, 0]
Ts = 1/fs  # sampling period
t = np.arange(samples)*Ts  # the times of sampling
x = t[indices]
ax.plot(x, y, label='clean', zorder=2)

# add white Gaussian noise
noise = np.random.randn(N)
noise *= (np.sum(y**2)/np.sum(noise**2)/10**(snr/10))**0.5
y += noise
ax.plot(x, y, label='with noise', zorder=1)

# add outliers
amp = 0.8*max(abs(y))
M = np.floor(percent_outlier*N).astype(int)
out_ind = np.random.choice(N, M)
outs = np.sign(np.random.randn(M))*amp
y[out_ind] += outs
ax.plot(x, y, label='with noise and outliers', zorder=0)

# finish figure
ax.set_xlabel('time in sec')
ax.set_ylabel('amplitude')
ax.legend()
ax.grid()

# unbiased L2 Kernel Ridge Regression (KRR-L2)
# build kernel matrix
pair_dist = np.abs(x.reshape(-1, 1) - x.reshape(1, -1)) # solution  
K = np.exp(-1/(sigma**2)*pair_dist**2) # solution  
A = C*np.identity(N) + K # solution  
sol = np.linalg.solve(A, y)

# Generate regressor
# NOTE: this loop can be optimized
z0 = np.zeros(samples)
for k in range(samples):
    z0[k] = 0
    for j in range(N):
        value = np.exp(-1/(sigma**2)*(x[j] - t[k])**2)
        z0[k] += sol[j]*value

# plot
fig, ax = plt.subplots()
ax.set_xlabel('time in sec')
ax.set_ylabel('amplitude')
ax.plot(x, y, label='input signal')
ax.plot(t, z0, label='reconstructed signal')
ax.legend()
ax.grid()

plt.show()