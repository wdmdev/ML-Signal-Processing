#%%
import numpy as np
import matplotlib.pyplot as plt
from convmtx import convmtx
from nlms import nlms
from rls import rls
from tqdm import tqdm

L = 200         # dimension of the unknown vector
N = 3500        # number of data samples
Nexp = 100      # number of experiments

# parameters for RLS
beta = 1.00     # forget factor
lambda_ = 1e-1  # regularization

# parameters for NLMS
mu = 1.2      # step-size
delta = 1e-3  # regularization

# parameters for the data generation process (data is generated according to
# the regression model)
# these refer to example 6.1
# y_n = theta^T * x + eta
theta = np.random.randn(L)
eta = 1e-2  # variance for the noise added to y

# reserve memory
E_RLS = np.zeros((N, Nexp))
E_NLMS = np.zeros((N, Nexp))

# iterate experiments
for i in tqdm(range(Nexp)):
    # generate normalized input
    X = np.random.randn(L, N)
    X = X / np.std(X,0)

    # create output
    noise = np.sqrt(eta)*np.random.randn(N)
    y = X.T@theta + noise
    # run RLS and NLMS
    _, E_RLS[:, i] = rls(X, y, L, beta, lambda_)
    _, E_NLMS[:, i] = nlms(X, y, L, mu, delta)


# plot
MSEav_rls = np.mean(E_RLS**2, axis=1)
MSEav_nlms = np.mean(E_NLMS**2, axis=1)

plt.figure(figsize=(14, 8))
plt.plot(10*np.log10(MSEav_nlms), color='b', label='NLMS')
plt.plot(10*np.log10(MSEav_rls), color='orange', label='RLS')
plt.legend(loc='upper right')
plt.xlabel('n')
plt.ylabel('mse [dB]')
plt.grid('on')
plt.show()
