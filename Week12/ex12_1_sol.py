#%%
import matplotlib.pyplot as plt
import numpy as np

#%% 12.1.1
# number of points
N = 100

# generate points for class 1
r = np.random.rand(N, 1)
T = 2*np.pi*np.random.rand(N, 1)
points = r*np.exp(1j*T)
class1 = np.hstack((np.real(points), np.imag(points)))
fig, ax = plt.subplots()
ax.grid()
ax.scatter(class1[:, 0], class1[:, 1], color='b')

# generate points for class 2
r = np.random.rand(N, 1) + 1.5
T = 2*np.pi*np.random.rand(N, 1)
points = r*np.exp(1j*T)
class2 = np.hstack((np.real(points), np.imag(points)))
ax.scatter(class2[:, 0], class2[:, 1], color='r')

# stack data matrix
X = np.vstack((class1, class2))
y = np.vstack((np.ones((1, N)), 2*np.ones((1, N))))

#%% 12.1.2
# apply the non-linear transform and plot in 3d to confirm that the points are
# now linearly seperable
PHI = np.vstack((X[:, 0]**2, 2**0.5*X[:, 0]*X[:, 1], X[:, 1]**2)).T # solution  
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(PHI[:N, 0], PHI[:N, 1], PHI[:N, 2], color='b')
ax.scatter(PHI[-N:, 0], PHI[-N:, 1], PHI[-N:, 2], color='r')

#%% 12.1.4
# calculate the homogenous polynomial kernel and plot
K = np.empty((2*N, 2*N))
r = 2
for i in range(2*N):
    for j in range(2*N):
        K[i, j] = (X[i, :] @ X[j, :])**r # solution  

# plot the K matrix on a color scale
fig, ax = plt.subplots()
ax.imshow(K)

#%% 12.1.5
# apply the representer theorem with theta_k=1 for all k
f = K.sum(axis=0) # solution  
fig, ax = plt.subplots()
ax.grid()
ax.scatter(np.arange(N), f[:N], color='b')
ax.scatter(np.arange(N, 2*N), f[-N:], color='r')
ax.imshow(K)

#%% 12.1.6
# calculate the Gaussian kernel and plot
sigma2 = 0.01 # sigma squared
K = np.empty((2*N, 2*N))
for i in range(2*N):
    for j in range(2*N):
        K[i, j] = np.exp(- np.linalg.norm(X[i, :]-X[j, :])**2 / (2*sigma2)) # solution  

# plot the K matrix on a color scale
fig, ax = plt.subplots()
ax.set_title('Gaussian Kernel')
ax.imshow(K)

# apply the representer theorem with theta_k=1 for all k
f = K.sum(axis=0)
fig, ax = plt.subplots()
ax.grid()
ax.scatter(np.arange(N), f[:N], color='b')
ax.scatter(np.arange(N, 2*N), f[-N:], color='r')
ax.imshow(K)
plt.show()