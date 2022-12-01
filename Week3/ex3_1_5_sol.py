#%% generate a signal
from gen import gen
from xcorr import xcorr
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.grid'] = True

N = 30
k = int(N/2)
a = 0.1
u = gen(N, a)
r0 = xcorr(u, u, k)

N = 30
k = int(N/2)
a = 0.9
u = gen(N, a)
r1 = xcorr(u, u, k)

fig = plt.figure(figsize=(12, 8))
uk = np.arange(-k,k+1)
plt.plot(uk, r0, color = 'b', label='N=30 a=0.1')
plt.plot(uk, r1, color = "r", label='N=30 a=0.9', linestyle = ":")
plt.legend(loc='upper right')
plt.show()

N = 1000
a = 0.1
k = int(N/2)
u = gen(N, a)
r2 = xcorr(u, u, k)

N = 1000
a = 0.9
k = int(N/2)
u = gen(N, a)
r3 = xcorr(u, u, k)

fig = plt.figure(figsize=(12, 8))
uk = np.arange(-k,k+1)
plt.plot(uk, r2, color = 'b', label='N=1000 a=0.1')
plt.plot(uk, r3, color = 'r', label='N=1000 a=0.9', linestyle = ":")

# solution
ar_cor = lambda a, k, sigma : (a ** np.abs(k))/(1-a**2)*sigma**2
k = np.arange(-500,501)

a = 0.1
plt.plot(k,ar_cor(a, k, 1), color = 'g', label='Theoretical AR(1)')
plt.legend(loc='upper right')
plt.show()
