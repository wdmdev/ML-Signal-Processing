#%%
from datetime import datetime, timedelta

import os
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
from hmmlearn.hmm import MultinomialHMM 

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# load data
T = pd.read_csv(os.path.join(DATA_DIR, 'Newly_admitted_over_time.csv'), sep=';')

# check sampling is consistent
strptime = lambda x: datetime.strptime(x, '%Y-%m-%d')
for i in range(len(T) - 1):
    diff = strptime(T.Dato[i+1]) - strptime(T.Dato[i])
    assert diff == timedelta(days=1)

# plot data
plt.figure()
plt.plot(T.Total)
plt.show()

#%%
# setup and pre-process
Y = T.Total
n_states = 3
mvAvg = 5
b = np.ones(mvAvg)/mvAvg
Y_MA = scipy.signal.lfilter(b, 1, Y)
Y = Y_MA

# create diff feature
diff_seq = Y[1:]-Y[:-1]

# change smaller than x is no change
diff_seq[abs(diff_seq) <= 1] = 0
diff_seq = np.sign(diff_seq)
Y = diff_seq + 1

# setup model
TRGUESS = np.array([
    [0.90, 0.10, 0.00],
    [0.10, 0.80, 0.10],
    [0.00, 0.10, 0.90],
])
EMITGUESS = np.array([
    [0.80, 0.15, 0.05],
    [0.05, 0.90, 0.05],
    [0.05, 0.15, 0.80],
])

# initialize HMM model
S = 3
tol = 1e-8
n_iter = 10**3
model = MultinomialHMM(n_components=S, n_trials=1, init_params='', tol=tol, n_iter=n_iter)
model.monitor_.verbose = True

# set initialization parameters
startprob_ = np.zeros(S)
startprob_[0] = 1
model.startprob_ = startprob_
model.transmat_ = TRGUESS
model.emissionprob_ = EMITGUESS
model.n_features = S

#%% fit model
# create one-hot encoding
Y_ = np.int_(Y.reshape(Y.size,))
Y_API = np.zeros((Y_.size, 3), dtype=int)
Y_API[np.arange(Y_.size), Y_] = 1
model.fit(Y_API)

# get the most likely states using viterbi
logprob, states = model.decode(Y_API)

# plot
fig, axes = plt.subplots(3, 1, figsize=(10, 6))
axes[0].plot(states)
axes[0].set_xlim(0, 250)
axes[1].plot(T.Total)
axes[1].plot(Y_MA)
axes[1].set_xlim(0, 250)
axes[2].plot(diff_seq+2)
axes[2].set_xlim(0, 250)

plt.show()