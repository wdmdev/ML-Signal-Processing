#%%
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Create 500 samples of the AR sequence
N = 500
x = np.zeros(N,)
sigma_n = 0.7
noise = sigma_n*np.random.randn(N,)
a = np.array([0.2, 0.1]) 

x[0] = 4
x[1] = -2

for n in range(2,N):
    x[n] = -a[0]*x[n-1] -a[1]*x[n-2] + noise[n]

# Create signal y = x + noise
sigma_v = 0.2 
noise_v = sigma_v * np.random.randn(N,)
y = x + noise_v

# Kalman Filter
F = np.array([[-a[0], -a[1]], [1, 0]])
H = np.array([[1, 0]])
Q = np.array([[np.square(sigma_n), 0], [0, 0]])
R = np.square(sigma_v) 
# initalization
x_hat = np.zeros(N,)
x0 = np.array([[0], [0]])

P = 0.01 * np.array([[1, 0], [0, 1]]) 
x_hat[0] = x0[-1]
x_hat[1] = x0[0]

# Iterations
# Naive implementation of the Kalman filter that computes inv(S). This 
# implementation is used so that the code resemblmes the math as close as
# possible.
# 
# This inversion should be avoided for a numerically stable algorithm, but
# it clutters the code.
for n in range(N-1):
    S  = R + H@P@H.T # solution  
    K  = P@H.T@inv(S) # solution  
    x0 = x0 + K@(y[n] - H@x0) # solution  
    P  = P - K@H@P # solution  
    x0 = F@x0
    P  = F@P@F.T + Q
    x_hat[n] = x0[-1]  # store values for next iteration
    x_hat[n+1] = x0[0] # store values for next iteration
# end of Kalman Filter

# plot the results
plt.figure(figsize=(10, 6))
plt.plot(np.arange(0,50), y[: 50], color="red")
plt.xlabel('n')
plt.ylabel('y_n')
plt.title("A realization of the observation sequence y_n, used by KF to obtain the predictions of the state variable.")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(np.arange(0,50), x[: 50], color="red", label = "state variable")
plt.plot(np.arange(0,50), x_hat[: 50], color="black", label = "prediction")
plt.xlabel('n')
plt.ylabel('x_n')
plt.legend()
plt.title("The AR process (state variable) in red together with the predicted by the KF sequence (black).")
plt.show()