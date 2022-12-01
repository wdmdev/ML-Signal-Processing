#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from scipy.fft import dct, idct

# Signal setup
N = 30  # number of observations to make
l = 2**8  # signal length
k = 3  # number of nonzero frequency components

a = [0.3, 1, 0.75]
posK = [4, 10, 30]

# Construct the multitone signal
x = np.zeros(l)
n = np.arange(l)
for i in range(k):
    x += a[i]*np.cos((np.pi*(posK[i]-0.5)*n)/l)

# From exercise 6.4.3
# Construct the sensing matrix 
positions = np.random.randint(l, size=N)
B = np.zeros((N, l))
for i in range(N):
    B[i, positions[i]] = 1
y = B@x

fig, ax = plt.subplots(1, 2, figsize=(12, 3))
ax[0].plot(x)
ax[0].plot(positions, y, 'r.')
ax[0].set_title('Original signal, time-domain')
ax[1].stem(idct(x, norm='ortho'))
ax[1].set_title('Original signal, IDCT-domain')
fig.tight_layout()

# Since it is sparse in the IDCT domain, i.e. B*x = B*Phi*X = BF*X,
# where X sparse,  BF = B*Phi; and Phi is the DCT matrix, Phi = dctmtx(l);.
# Equivalently, since IDCT = transpose of DCT using idct we can write
BF = idct(B, norm='ortho')

lambda_ = 0.005
model = Lasso(lambda_, fit_intercept=False)
model.fit(BF, y)
solsB = model.coef_

# create OMP solution
k_OMP = 4  # number of vectors
X = BF  # set X
# initialize
residual = y
S = np.zeros(k_OMP, dtype=int)
normx = np.sqrt(np.sum(X**2, axis=0)) # shortcut formula
for i in range(k_OMP):
    proj = X.T@residual/normx  # solution  
    pos = np.argmax(abs(proj))
    S[i] = pos
    Xi = X[:, S[:i+1]]
    theta_ = np.linalg.solve(Xi.T @ Xi, Xi.T @ y) # solution  
    theta = np.zeros(X.shape[1])
    theta[S[:i+1]] = theta_
    residual = y - X@theta # solution  
solsOMP = theta

# plot solutions
fig, ax= plt.subplots(1, 1, figsize=(6, 3))
ax.stem(solsB, markerfmt='bo', label='sklearn Lasso', basefmt=' ')
ax.stem(solsOMP, markerfmt='ro', label='OMP', basefmt=' ')
ax.legend()
ax.set_title('Solutions')

# Take the inverse IDCT (i.e. the DCT) in order to compute the estimated signal. 
x_hat = dct(solsOMP, norm='ortho')

fig, ax= plt.subplots(1, 2, figsize=(12, 3))
ax[0].plot(x)
ax[0].set_title('Original signal')
ax[1].plot(x_hat)
ax[1].set_title('Estimated using randomly picked samples')
fig.tight_layout()
plt.show()