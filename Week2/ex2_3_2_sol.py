#%% Parameters
import numpy as np
import matplotlib.pyplot as plt

theta_t = np.array([[1], [2], [0.5]]) # True weights
noiselevel = 1.0      # Standard deviation of Gaussian noise on data
d = len(theta_t)      # Number of dimensions
N = 5                 # Training set size
Ntest = 1000          # Size of test set 
repetitions = 100     # number of repetitions
lambda_min =   0.03   # minimal weigth decay
lambda_max = 100.0    # maximal weigth decay
M = 100               # number of weight decays

# Pre-allocation of variables
lambdas = M*[None]
meanarr = M*[None]
biasarr = M*[None]
variancearr = M*[None]

Ytest = np.zeros((Ntest, repetitions))

# Make statistical sample of samples for bias&variance
np.random.seed(42)

n = 0
print("Number of weight decays is ", str(M))
for k in range(M):
    lambda_ = lambda_min * (np.power((lambda_max/lambda_min), k/(M-1)))
    
    # d-dimensional model data set
    Xtest  = np.random.randn(Ntest, d) 
    Xtest[:,0] = 1
    Ttest = np.matmul(Xtest, theta_t)
    Ttest = Xtest @ theta_t
    noisetest = np.random.randn(Ntest, 1) * noiselevel
    Ttest = Ttest + noisetest
    
    for j in range(repetitions):
        # Small model (d-1) dimensional        
        Xtrain=np.random.randn(N,d)
        Xtrain[:,0] = 1
        Ttrain = Xtrain @ theta_t
        noise = np.random.randn(N,1) * noiselevel
        Ttrain = Ttrain + noise
        
        # Find optimal weights for the regularized model        
        A = Xtrain.T @ Xtrain + lambda_*np.eye(d)  # solution
        b = Xtrain.T @ Ttrain                      # solution
        theta = np.linalg.solve(A,b)

        # compute test set predictions
        Ytest[:,j] = np.squeeze(Xtest @ theta)
        
    Ybias = np.mean(Ytest,axis=1)
    Ybias = Ybias[:, np.newaxis]
    bias_error = np.mean(np.square(Ybias-Ttest))
    mean_error=0

    for j in range(repetitions):
        mean_error = mean_error + np.mean(np.square(Ytest[:,j,np.newaxis]-Ttest))
    
    mean_error=mean_error/repetitions
    variance_error=mean_error-bias_error  

    lambdas[n] = lambda_ 
    meanarr[n]=mean_error
    biasarr[n]=bias_error
    variancearr[n]=variance_error
    n=n+1
print("Weight decay # ", str(k+1), " of ", str(M), " decays done.")

# Plot results
fig = plt.figure(figsize=(10, 8))
plt.semilogx(lambdas, meanarr, label='mean test error', color = "red")
plt.semilogx(lambdas, biasarr, label='bias', color = "blue")
plt.semilogx(lambdas, variancearr, label='variance', color = "green")

plt.xlabel("weight decay")
plt.ylabel("mean square errors (test, bias and variance)")

plt.legend(loc='upper right')
plt.grid()
plt.show()