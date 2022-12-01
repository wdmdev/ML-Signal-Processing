#%%
# 10.1.2
import numpy as np
from hmmlearn.hmm import MultinomialHMM 

'''
If you get import error, try
pip install hmmlearn
pip install pandas

This script should be able to run without any failed assertions.
If any assert fails, then you made an error

Try your best and fix the error without looking in the solution, since you can
simply check the validity of your implementation using asserts
'''

tol = 1e-10

# create HMM model according to 16.5.1 steps
# probabilities for initial state
P_k = np.array([0.7, 0.3])

# transition probabilities
P_ij = np.array([
    [0.9, 0.1],
    [0.2, 0.8],
])

# state emission distributions, we use a discrete distribution
p_y = np.array([
    [0.2, 0.6, 0.1, 0.1],
    [0.1, 0.2, 0.5, 0.2],
])

# store number of states, and size of discrete distribution
nstates = len(P_k)
nemission = p_y.shape[1]

# calculate p(y_1) (all possible combinations), should be size 1 x nemission
p_y_1 = p_y.T@P_k  # solution  

# calculate p(y_1) using hmmlearn module to validate calculations
model = MultinomialHMM(n_components=nstates, n_trials=1)
model.startprob_ = P_k
model.transmat_ = P_ij
model.emissionprob_ = p_y
model.n_features = nemission

p_y_1_ = np.empty(nemission)
for y_1 in range(nemission):
    # API assumes one-hot encoding
    Y = np.zeros((model.n_trials,nemission))
    Y[0][y_1] = 1
    logprob = model.score(Y)
    p_y_1_[y_1] = np.exp(logprob)

# check that own implementation gives the same as library
assert np.all(np.linalg.norm(p_y_1 - p_y_1_) < tol)

#%%
# 10.1.3
p_y1_y2 = np.zeros((nemission, nemission))
p_y1_y2_ = np.zeros((nemission, nemission))
#model.n_trials = 2

# do naive calculation of p(y1,y2) using marginalization (all possible
# combinations)
for y_1 in range(nemission):
    # API assumes one-hot encoding, create Y as one-hot
    Y = np.zeros((2,nemission))
    Y[0][y_1] = 1
    for y_2 in range(nemission):
        Y[1] = np.zeros((1,nemission))
        Y[1][y_2] = 1
        P = 0
        for x_1 in range(nstates):
            for x_2 in range(nstates):
                P = P + p_y[x_2, y_2]*P_ij[x_1, x_2]*p_y[x_1, y_1]*P_k[x_1]  # solution  
        p_y1_y2[y_1, y_2] = P

        # get probability of sequence from HMM library
        #Y = np.array([[y_1], [y_2]])
        logprob = model.score(Y)
        p_y1_y2_[y_1, y_2] = np.exp(logprob)

# check that our manual calculation sums to one
assert np.abs(p_y1_y2_.sum() - 1) < tol

# check that own implementation gives the same as library
assert np.all(np.linalg.norm(p_y1_y2_ - p_y1_y2) < tol)

#%%
# 10.1.8
# given a specific sequence, calculate P(x_n|y_[1_n])
y = np.array([0, 1, 1, 3, 0])

alpha = np.empty((len(P_k), len(y)))
alpha[:, 0] = P_k*p_y[:, y[0]]  # solution  
# implement formula 16.43
for i in range(1, len(y)):
    alpha[:, i] = p_y[:, y[i]] * np.sum(alpha[:, i-1]*P_ij.T, axis=1)  # solution  

# validate that calculation is correct
p_Y = np.sum(alpha[:, -1])
if len(y) == 2:
    assert p_y1_y2(y[0], y[1]) - np.sum(alpha[:, 1]) < tol

# implement formula for P(x_n|y_[1_n]) (below 16.43)
P_x_n = alpha[:, -1]/p_Y # solution  

# validate that calculation is correct using library function
# create Y_api as one-hot encoding
Y_ = np.int_(y.reshape(y.size,))
Y_API = np.zeros((Y_.size, nemission), dtype=int)
Y_API[np.arange(Y_.size), Y_] = 1

# get posteriors
_, posteriors = model.score_samples(Y_API)

# error check
assert np.linalg.norm(P_x_n - posteriors[-1]) < tol