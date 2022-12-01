#%%
import numpy as np
from sklearn.linear_model import LassoLars
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

l = 2**8  # signal length  
k = 3  # number of non-zero frequency components   

a = [0.3, 1, 0.75]  # components amplitude
posK = [4, 10, 30]  # components location

# Construct the multitone signal
x = np.zeros(l)
n = np.arange(l)
for i in range(k):
   x += a[i]*np.cos((np.pi*(2*posK[i]-1)*n)/(2*l))

x_k = idct(x)/l # normalize as specified by DCT

plt.figure(1, figsize=(15,5))
plt.subplot(2, 1, 1)
plt.plot(x)
plt.title('Signal')
plt.subplot(2, 1, 2)
plt.stem(x_k, basefmt=' ')
plt.title('DCT domain')
plt.show()

#%%
# Ex 6.4.2
# Construct the sensing matrix with variance N(0, 1/N)
N = 30  # number of observations to make  
A = np.random.randn(N, l)/N**0.5  
y = A@x

# Since it is sparse in the IDCT domain, i.e. A*theta = A*Phi*X = AF*X,
# where X sparse,  AF = A*Phi and Phi is the DCT matrix, Phi = dctmtx(l);.
# Equivalently, using idct (for faster computation than with the DCT matrix), AF is computed as:
AF = idct(A, axis=1)

# Use LassoLars
model = LassoLars(alpha=0.005, fit_intercept=False, normalize=False,  max_iter=1e6)
model.fit(AF, y)
solsA = model.coef_

# Take the IDCT (i.e. the DCT) in order to compute the estimated signal.
x_hat = dct(solsA)

# Plot
plt.figure(2, figsize=(15, 5))
plt.subplot(2, 1, 1)
plt.plot(x)
plt.title('Original')
plt.subplot(2, 1, 2)
plt.plot(x_hat)
plt.title('Estimated (random sensing matrix)')
plt.show()

#%%
# Exercise 6.4.3
# Construct the sensing matrix
positions = np.random.choice(l, N, replace=False)
B = np.zeros(shape=(N, l))
for i in range(0, N):
    B[i, positions[i]] = 1

y = B@x

plt.figure(3, figsize=(15, 5))
plt.plot(x)
plt.plot(positions, y, 'r.')
plt.title('Samples taken')

# Since it is sparse in the IDCT domain, i.e. B*theta = B*Phi*X = BF*X,
# where X sparse,  BF = B*Phi; and Phi is the DCT matrix, Phi = dctmtx(l);.
# Equivalently, using idct (for faster computation than with the DCT matrix), AF is computed as:
BF = idct(B, axis=1)  

# Use LassoLars
model = LassoLars(alpha=0.01, fit_intercept=False, normalize=False,  max_iter=1e6)
model.fit(BF, y)
solsB = model.coef_
print(f'Number of non-zero weights: {np.count_nonzero(solsB)}')

# Take the inverse IDCT (i.e. the DCT) in order to compute the estimated signal.
x_hat = dct(solsB, axis=0)  

plt.figure(4, figsize=(15,5))
plt.subplot(2, 1, 1)
plt.plot(x)
plt.plot(positions, y, 'r.')
plt.title('Original + Samples taken')
plt.subplot(2, 1, 2)
plt.plot(x_hat)
plt.title('Estimated (using randomly picked samples)')

plt.show()
