#%%
import numpy as np
from numpy import linalg as LA
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

N = 20 # number of data samples  
K = 5  # number of non-zero weights  
l = 50  # total number of weightts  

lambda_lasso = 1e-4
lambda_ridge = 1e-4

theta = np.zeros(l)
theta[:K] = np.random.randn(K)

X = np.random.randn(N, l)/N**0.5  
y = X@theta  

lasso = Lasso(alpha=lambda_lasso, fit_intercept=False)
lasso.fit(X, y)

sols_lasso = lasso.coef_
error_lasso = LA.norm(sols_lasso-theta)

print("Euclidan norm from difference between predicted and original weights for Lasso: ", error_lasso)
# ---------------------------------------------------------------------------------------------------
ridge = Ridge(alpha=lambda_ridge)
ridge.fit(X, y) 

sols_ridge = ridge.coef_
error_ridge = LA.norm(sols_ridge-theta)
print("Euclidan norm from difference between predicted and original weights for Ridge: ", error_ridge)
# ---------------------------------------------------------------------------------------------------
plt.figure(figsize=(16,10))
plt.stem(range(l), theta, linefmt = 'blue', markerfmt='bo', label='true vector', basefmt=" ", use_line_collection=True)
plt.stem(range(l), sols_lasso,  linefmt = 'none', markerfmt='r.', label='lasso', basefmt=" ", use_line_collection=True)
plt.stem(range(l), sols_ridge,  linefmt = 'none', markerfmt='g.', label='ridge', basefmt=" ", use_line_collection=True)
plt.legend()
plt.grid()
plt.show()