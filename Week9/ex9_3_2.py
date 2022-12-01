#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
np.random.seed(48)

# true signal curve
x = np.arange(0, 2, 0.1e-4) # Start = 0, Stop = 2, Step Size = 0.1e-4
y = 0.2 * np.ones(np.size(x, axis = 0),) - x + 0.9 * x**2 + 0.7 * x**3 - 0.2 * x**5

# training samples (20 or 500)
N = 500

# linear coefficients
K = 5

# sample interval [a b]
a = 0 
b = 2

# generate samples
x1 = np.arange(a, b, b/N)

# noise generation 
sigma_eta = 0.05
n = np.sqrt(sigma_eta) * np.random.randn(N,) 

# use the true theta 
theta_true = np.array([.2, -1, .9, .7, -.2]) 

# compute the measurement matrix
Phi = np.array([np.ones(N,), x1, x1**2, x1**3, x1**5]).T
Phi_gram = Phi.T @ Phi

# generate noisy observations using the linear model
y1 = Phi @ theta_true + n

# EM algorithm 
# initializate parameters
# experiment on different initializations
EMiter = 20
betaj = 1
sigma_eta_EM = np.ones(EMiter,)
alphaj = 1
Phiy = Phi.T @ y1

for i in range(EMiter):
    Sigma_theta = np.linalg.inv(betaj * Phi_gram + alphaj * np.eye(K))
    mu_theta = betaj * Sigma_theta @ Phiy
    
    alphaj = K/(np.square(np.linalg.norm(mu_theta)) + np.trace(Sigma_theta))
    
    betaj = N / (np.square(np.linalg.norm(y1 - Phi @ mu_theta)) + np.trace(Sigma_theta @ Phi_gram))
    sigma_eta_EM[i] = 1/betaj


# perform prediction on new samples
Np = 10

# generate prediction samples
x2 = (b-a) * np.random.rand(Np,)

# compute prediction measurement matrix 
Phip = np.column_stack((np.ones(Np,), x2, x2**2, x2**3, x2**5))

# compute the predicted mean and variance
y_pred = Phip @ mu_theta
y_pred_var = np.diagonal(sigma_eta_EM[-1] + sigma_eta_EM[-1] * 1/alphaj * Phip @ np.linalg.solve(sigma_eta_EM[-1] * np.eye(K) + 1/alphaj * Phi_gram, Phip.T))

# plot the predictions along the condifence intervals
matplotlib.rcParams.update({'errorbar.capsize': 4})
plt.figure(figsize=(14, 10))
plt.plot(x, y, color="black")
plt.plot(x2, y_pred, marker='x', linestyle='', color='r')
plt.errorbar(x2, y_pred, yerr = y_pred_var, linestyle='', fmt='.r')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Number of samples: %i " %N)
plt.show()

# plot the noise variance throughout iterations
plt.figure(figsize=(14, 10))
plt.plot(np.arange(EMiter), sigma_eta_EM, color="black")
plt.axhline(y=sigma_eta, color='r', linestyle='-')
plt.axis((0,EMiter,0.048, np.max(sigma_eta_EM)))
plt.xlabel('Iterations')
plt.ylabel('Noise variance')
plt.title("Number of samples: %i " %N)
plt.show()