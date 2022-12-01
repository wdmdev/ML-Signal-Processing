#%%
import numpy as np
import matplotlib.pyplot as plt
from convmtx import convmtx
from nlms import nlms
from rls import rls
from tqdm import tqdm

L = 5           # dimension of the unknown vector
N = 1000        # number of data samples
Nexp = 200      # number of experiments

# parameters for RLS
beta = 0.995    # forget factor
lambda_ = 1e-3  # regularization

# parameters for NLMS
mu = 0.5      # step-size
delta = 1e-3  # regularization

# parameters for the data generation process
# these refer to eq (6.61)
alpha = 0.97  # autoregressive coefficient for theta changes
omega = 1e-2  # variance for the noise added to theta
eta = 1e-3  # variance for the noise added to y

# reserve memory
E_RLS = np.zeros((N, Nexp))
E_NLMS = np.zeros((N, Nexp))

# iterate experiments
for i in tqdm(range(Nexp)):
    # draw thetas
    theta = np.zeros((L, N))
    theta[:, 0] = np.random.randn(L)

    # Generate the time varying channel
    for j in range(1, N):
        theta[:, j] = alpha*theta[:, j-1] + np.sqrt(omega)*np.random.randn(L)

    # generate the X and y's
    X = np.random.randn(L, N)
    noise = np.random.randn(N) * np.sqrt(eta)

    y = np.sum(X*theta,0)
    y = y + noise

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
