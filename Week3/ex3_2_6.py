#%%
import numpy as np
a = 0.6
sigma_eta = 0.8
sigma_epsilon = 1 
sigma_s = np.sqrt(sigma_eta**2/(1-a**2))
Gamma_ss = np.array([[1, a, a**2], [a, 1, a], [a**2, a, 1]]) * sigma_s**2
Gamma_epsilon = np.eye(3) * np.square(sigma_epsilon)
Gamma_l = Gamma_ss+Gamma_epsilon
gamma_ss = np.array([[1], [a], [a**2]]) * sigma_s**2
w = np.linalg.solve(Gamma_l, gamma_ss) # Solve systems of linear equations Ax = B for x
print(w)