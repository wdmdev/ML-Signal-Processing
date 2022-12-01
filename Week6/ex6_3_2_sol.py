#%%
import numpy as np
from numpy import linalg as LA
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

N = 20  # number of data samples
K = 5  # number of non-zero weights
l = 50  # total number of weightts
rep = 100  # number of repetitions

lambda_lasso = 1e-4
lambda_ridge = 1e-4

theta  = np.zeros(l)
theta[:K] = np.random.randn(K)

Err_lasso = np.zeros(rep)
Err_ridge = np.zeros(rep)
Err_lasso_p = np.zeros(rep)
Err_ridge_p = np.zeros(rep)

for epan in range(rep):  # we randomize data and fit a model for each repetition
    X = np.random.randn(N, l)/N**0.5
    y = X@theta

    lasso = Lasso(alpha=lambda_lasso, max_iter=10000, fit_intercept=False)
    lasso.fit(X, y)
    sols_lasso = lasso.coef_
    error_lasso = LA.norm(sols_lasso-theta)
    Err_lasso[epan] = error_lasso
    Err_lasso_p[epan] = error_lasso < 0.01  # the reconstruction is assumed correct if the error is < 0.01
    ridge = Ridge(alpha=lambda_ridge)
    ridge.fit(X, y) 
    sols_ridge = ridge.coef_
    error_ridge = LA.norm(sols_ridge-theta)
    Err_ridge[epan] = error_ridge
    Err_ridge_p[epan] = error_ridge < 0.01  # the reconstruction is assumed correct if the error is < 0.01
# ---------------------------------------------------------------------------------------------------    
probrandn_lasso = np.sum(Err_lasso_p)/rep   
probrandn_ridge = np.sum(Err_ridge_p)/rep   

print('Probability of correct reconstruction for lasso: ', probrandn_lasso)
print('Probability of correct reconstruction for ridge: ', probrandn_ridge)
# ---------------------------------------------------------------------------------------------------
plt.figure(figsize=(16,10))
plt.plot(Err_lasso, label='lasso')
plt.plot(Err_ridge, label='ridge')
plt.ylabel('error')
plt.xlabel('repetition')
plt.legend()
plt.grid()
plt.show()