#%% generate a signal
from xcorr import xcorr
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
plt.rcParams['axes.grid'] = True

np.random.seed(42)
eta = 0.8 * np.random.normal(size=1000)
epsilon = np.random.normal(size=1000)

fig = plt.figure()
plt.plot(eta)
plt.show()

fig = plt.figure()
plt.plot(epsilon)
plt.show()

# Estimate correlation functions
a = 0.6
s = signal.lfilter([1], [1,-a], eta)

u = s + epsilon
r_uu = xcorr(u, u, 2)
r_du = xcorr(s, u, 2)
R_uu = np.vstack((r_uu[2:5], r_uu[1:4], r_uu[0:3]))

w_hat = np.linalg.solve(R_uu, r_du[2:5])

print("R_xx: ", R_uu)
print("hhat: ", w_hat)