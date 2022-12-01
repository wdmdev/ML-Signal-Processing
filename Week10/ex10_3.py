#%%
import numpy as np
from hmmlearn.hmm import MultinomialHMM 

reps = 1  # repetitions of random initialization
n_iter = 10**4  # max no. of EM updates
tol = 1e-12  # tolerance
N = 100
S = 2  # no of states in data generation
K = 4  # dimension of observation space
Psss = 0.99  # probability to stay in the same state
Pses = 0.95  # probability of having the same emmision as the state

# generate pseudo random transition matrix where prob of staying in the same state is Psss
P_ij = Psss*np.identity(S)
for i in range(S):
    a = np.random.rand(S-1)
    a = (1-Psss)*a/a.sum()
    P_ij[P_ij[:, i] == 0, i] = a

# generate pseudo random emmision matrix
sB = min(K, S)
p_y = np.zeros((K, S))
for i in range(sB):
    p_y[i, i] = Pses
for i in range(S):
    if i > K:
        a = np.random.rand(K)
    else:
        a = np.random.rand(K-1)
    a = (1-Pses)*a/a.sum()
    p_y[p_y[:, i] == 0, i] = a

p_y = p_y/p_y.sum(axis=0)

# set starting probabilities
startprob_ = np.zeros(S)
startprob_[0] = 1

# initialize an HMM process that will generate the samples
model_gen = MultinomialHMM(n_components=S, n_trials=1, init_params='')
model_gen.startprob_ = startprob_
model_gen.transmat_ = P_ij
model_gen.emissionprob_ = p_y.T
model_gen.n_features = K

# generate the samples
Y, states = model_gen.sample(N)

# initialize another model to train with the EM algorithm
#model_train = HMM(n_components=S, init_params='', tol=tol, n_iter=n_iter)
model_train = MultinomialHMM(n_components=S, n_trials=1, init_params='', tol=tol, n_iter=n_iter)
model_train.monitor_.verbose = True

# set EM algorithm initialization parameters
model_train.startprob_ = startprob_
model_train.transmat_ = P_ij
model_train.emissionprob_ = p_y.T
model_train.n_features = K

# fit
_ = model_train.fit(Y)

# use random initialization
# loop reps times to complete these lines
# for M models, train HMM using
# Pij_init = P_ij;
# Py_k_init = p_y;
# Then train M models using random init