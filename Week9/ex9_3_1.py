#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# true signal curve
x = np.arange(0, 2, 0.1e-4) # Start = 0, Stop = 2, Step Size = 0.1e-4
y = 0.2 * np.ones(np.size(x, axis = 0),) - x + 0.9 * x**2 + 0.7 * x**3 - 0.2 * x**5

# training samples (20 or 500)
N = 20

# sample interval [a b]
a = 0 
b = 2

# generate samples
x1 = np.arange(a, b, b/N)

# noise generation 
sigma_n = 0.05
n = np.sqrt(sigma_n) * np.random.randn(N,) 

# use the true theta 
theta_true = np.array([.2, -1, .9, .7, -.2]) 

# or a random one
theta_dstrbd = np.array([-0.004, -10.54, 0.465, 0.087, -.093])
l = np.size(theta_true, axis = 0)

# compute the measurement matrix
Phi = np.array([np.ones(N,), x1, x1**2, x1**3, x1**5]).T

# generate noisy observations using the linear model
y1 = np.dot(Phi, theta_true) + n

# set the parameters of Gaussian prior
sigma_theta = 2
mu_theta_prior = theta_true # or mu_theta_prior = theta_dstrbd;

# compute the precision matrix of the Gaussian posterior
Sigma_theta_pos = sigma_theta**-1 * np.eye(l) + sigma_n**-1 * Phi.T @ Phi

# compute the posterior mean
mu_theta_pos =  mu_theta_prior + sigma_n**-1 * np.linalg.solve(Sigma_theta_pos, Phi.T) @ (y1 - Phi @ mu_theta_prior)

# linear prediction
Np = 20 

# generate prediction samples
x2 = (b-a) * np.random.rand(Np,)

# compute prediction measurement matrix 
Phip = np.array([np.ones(Np,), x2, np.power(x2, 2), np.power(x2, 3), np.power(x2,5)]).T

# compute the predicted mean and variance
mu_y_pred = Phip @ mu_theta_pos
inv_mat = sigma_n * np.eye(l) + sigma_theta * Phi.T @ Phi
sigma_y_pred = np.diagonal(sigma_n + sigma_n * sigma_theta * Phip @ np.linalg.solve(inv_mat, Phip.T))

# plot the predictions along the confidence intervals
matplotlib.rcParams.update({'errorbar.capsize': 4})
plt.figure(figsize=(14, 10))
plt.plot(x, y, color="black")
plt.plot(x2, mu_y_pred, marker='x', linestyle='', color='r')
plt.errorbar(x2, mu_y_pred, yerr = sigma_y_pred, linestyle='', fmt='.r')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Number of samples: %i " %N)
plt.show()