#%% generate a signal
from gen import gen
from xcorr import xcorr
from crosscor_sol import crosscor
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.grid'] = True

N = 50
a =.1
k = 25 
u = gen(N,a)

uk = np.arange(-k,k+1,1)
[r_uu, mval] =crosscor(u,u,k)
r_uu_ref = xcorr(u,u,k) # reference implementation
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(uk, r_uu, label='$r_u(k)$')
ax.plot(uk, r_uu_ref, label='$r_u(k)$ reference')
ax.set_xlabel('$k$')
ax.set_ylabel('$r(k)$')
ax.legend()
plt.show()

# check that implementation calculates the correct down to tolerence
# if the assert fails your implementation is incorrect
tol = 1e-10
assert(np.all(np.abs(r_uu-r_uu_ref) < tol))