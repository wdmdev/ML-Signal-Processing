#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from ICA import ICA

# generate data
N = 5000
r = np.random.rand(N,2)

A = [[2, 1], [1, 1]] # solution  
x = (A@r.T).T # solution  

# plot generated data
plt.figure(figsize=(10,10))
plt.plot(x[:,0],x[:,1],'.')
plt.show()

# calculate pca
pca = PCA(n_components= np.size(x, 1))
pca.fit(x)  

U = pca.components_ # principal components
V = pca.transform(x)

# plot data projection on principal axis
plt.figure(figsize=(10,10))
plt.plot(V[:,0],V[:,1],'.')
plt.show()

# calculate ica
mu = 0.1
components = 2
iterations = 200

# Mean across the first (column) axis
col_means = np.mean(x, axis=0)
x = x - col_means

# run ICA
W = ICA(x, mu, components, iterations, 'subGauss')

# Normalize unmixing matrix
W = np.divide(W, np.max(W))

# Compute unmixed signals
y = (W@x.T).T

# plot data projection on ica axis
plt.figure(figsize=(10,10))
plt.plot(y[:,0],y[:,1],'.')
plt.show()