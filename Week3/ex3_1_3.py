#%%
from gen import gen
from xcorr import xcorr
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.grid'] = True

N = 100
Fcut = 0.25
d = 3
c1 = 0.3
c2 = 0.7

# generate artificial data 
x = gen(N, 0.5)
y = signal.lfilter(signal.firwin(30,Fcut), 1, x)

# generate u
u = np.zeros(N-d,)
for n in range(d, N):
    u[n-d] = c1 * x[n] + c2*y[n-d]
   
# est
r_uu_est = xcorr(u, u, N-d-1)

# calc
r_xx = xcorr(x,x,N-1)
r_yy = xcorr(y,y,N-1)
r_xy = xcorr(x,y,N-1)

r_uu = np.zeros((2*(N - d)-1))

for k in range(N-d):
    r_uu[k] =  c1**2*r_xx[k+N-1]+c2**2*r_yy[k+N-1]+c1*c2*r_xy[d+k+N-1]+c1*c2*r_xy[d-k+N-1]

r_uu = np.concatenate((np.flipud(r_uu[:N-d]), r_uu[1:N-d]))

fig = plt.figure(figsize=(12, 8))

uk = np.arange(-N+d+1,N-d)

plt.plot(uk, r_uu_est, color = 'b', label='Est')
plt.plot(uk, r_uu, color = "r", label='Calc')
plt.legend(loc='upper right')
plt.show()