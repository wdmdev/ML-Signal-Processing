#%%
import numpy as np
import matplotlib.pyplot as plt

def DFT(x):
    N = len(x)
    n = np.arange(N).reshape(-1, 1)
    k = np.arange(N).reshape(1, -1)

    # create the projection matrix
    PhiH = np.exp(-2*1j*np.pi*n@k/N) # solution  
    X = PhiH@x  # make the DFT projection
    return X


Fs = 100  # sampling frequency
Ts = 1/Fs  # Sampling interval
tend = 0.3  # Signal duration
t = np.arange(0, tend, Ts)

ff = 10  # Frequency of the signal
x = np.sin(2*np.pi*ff*t)
N = len(x)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
# Plot discrete signal
ax[0].stem(t, x)
ax[0].plot(t, x, 'b--')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[0].set_title('Original sinusoid')
ax[0].grid()

# determine the DFT of the signal
X = DFT(x)
X = X[:N//2+1]

# create frequency axis
frq = np.arange(N)*Fs/N  # Two sides frequency range
frq = frq[:N//2+1]  # One side frequency range
ax[1].stem(frq, abs(X))
ax[1].set_xlabel('Frequency (Hz)')
ax[1].set_ylabel('|X(freq)|')
ax[1].grid()

fig.tight_layout()
plt.show()