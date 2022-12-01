#%%
import numpy as np
import scipy.signal as sps
import scipy.linalg
import matplotlib.pyplot as plt
from ICA import ICA

np.random.seed(42)
n = 1000
fs = 100
noise = 3

# simulate EEG data with eye blinks
t = np.arange(n)
alpha = np.abs(np.sin(10 * t / fs)) - 0.5
alpha[n//2:] = 0
blink = np.zeros(n)
blink[n//2::200] += -1
blink = sps.lfilter(*sps.butter(2, [1*2/fs, 10*2/fs], 'bandpass'), blink)

frontal = blink * 200 + alpha * 10 + np.random.randn(n) * noise
central = blink * 100 + alpha * 15 + np.random.randn(n) * noise
parietal = blink * 10 + alpha * 25 + np.random.randn(n) * noise

eeg = np.stack([frontal, central, parietal]).T  # shape = (100, 3)

# decompose EEG using our ICA implementation and plot components
I = 300 # Number of iterations
num_components = 3 # Number of components

W = ICA(eeg, 0.1, num_components, I, 'superGauss')

# Normalize unmixing matrix
W = np.divide(W, np.max(W))

# Compute unmixed signals
y = (W@eeg.T).T

# plot original data
plt.figure(figsize=(14, 10))

plt.subplot(3, 1, 1)
plt.plot(frontal + 50)
plt.plot(central + 100)
plt.plot(parietal + 150)
plt.yticks([50, 100, 150], ['Fz', 'Cz', 'Pz'])
plt.ylabel('original data')

# decompose EEG and plot components
W = ICA(eeg, 0.05, num_components, I, 'superGauss')
components = (W @ eeg.T).T

plt.subplot(3, 1, 2)
plt.plot([[np.nan, np.nan, np.nan]])  # advance the color cycler to give the components a different color :)
plt.plot(components + [0.5, 1.0, 1.5])
plt.yticks([0.5, 1.0, 1.5], ['0', '1', '2'])
plt.ylabel('components')

# looks like component #3 contains the eye blinks
# let's remove them (hard coded)!
components[:, 2] = 0

# reconstruct EEG without blinks
x_reconstruct = scipy.linalg.solve(W, components.T).T

plt.subplot(3, 1, 3)
plt.plot(x_reconstruct + [50, 100, 150])
plt.yticks([50, 100, 150], ['Fz', 'Cz', 'Pz'])
plt.ylabel('cleaned data')