#%%
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from random import randrange
from sklearn.svm import SVR

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# function for adding white gaussian noise with respect to SNR (signal to noise ratio), source: https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python
def awgn(signal, snr):
    x_watts = signal ** 2
    # Set a target SNR
    target_snr_db = snr
    # Calculate signal power and convert to dB 
    sig_avg_watts = np.mean(x_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    # Generate an sample of white noise
    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(x_watts))
    # Noise up the original signal
    y_volts = signal + noise_volts    
    return y_volts

# parameters
N=100
samples = 1000
snr = 80 #dB
percent_outlier = 0.1

# learning parameters
epsilon=0.003
kernel_type='Gaussian'
kernel_params=0.004
C=1

# load bladerunner data
indices = np.arange(0, 1000, 10) 
start = 100000
sound, fs = librosa.load(os.path.join(DATA_DIR, 'BladeRunner.wav'),mono=False, sr = 44100)
sound = sound[:, start-1 : start +samples]
y = sound[0, indices]
Ts = 1/fs # h sampling period
t = np.arange(0,samples+1) * Ts # the times of sampling
x = t[indices]

# Add white Gaussian noise
y_noised = awgn(y, snr)

#Add outliers
O = 0.8 * np.max(abs(y_noised))
M = int(np.floor(percent_outlier * N))
out_ind = np.zeros(M,)
for i in range(M):
    out_ind[i] = randrange(N)
    
out_ind = out_ind.astype(int)
    
outs = np.sign(np.random.randn(M,1)) * O
y_noised[out_ind] = y_noised[out_ind] + np.squeeze(outs)

# convert data to proper dimensions in order to fit requirements of the library
x_col = x.reshape(( np.size(x), 1))
y_row = np.copy(y_noised)
t_col = t.reshape(( np.size(t), 1))

t_col = np.around(t_col, decimals=4)
x_col = np.around(x_col, decimals=4)
y_row = np.around(y_row, decimals=4)

plt.figure(figsize=(16,10))
plt.subplot(2,1,1)
plt.stem(x,y,  linefmt = 'none', markerfmt='k.', basefmt=" ", use_line_collection=True)
plt.title("Original y")
plt.xlabel("Time in (s)")
plt.ylabel("Amplitude")
plt.subplot(2,1,2)
plt.stem(x_col,y_row,  linefmt = 'none', markerfmt='k.', basefmt=" ", use_line_collection=True)
plt.title("y with added noise and outliers")
plt.xlabel("Time in (s)")
plt.ylabel("Amplitude")
plt.show()

# ---------- Support Vectore Regression -----------
gamma = 1/(np.square(kernel_params)) # gamma needs to be calculated in order to use 'Gaussian' kernel, which is not available in the library
regressor = SVR(kernel='rbf', gamma=gamma, C=C, epsilon=epsilon)

regressor.fit(x_col,y_row)
y_pred = regressor.predict(t_col)

# plot
plt.figure(figsize=(16,10))
plt.stem(x_col[regressor.support_], y_row[regressor.support_], linefmt = 'none', markerfmt='yo', label='support vector', basefmt=" ", use_line_collection=True)
plt.stem(x_col, y_row,  linefmt = 'none', markerfmt='k.', label='noised values', basefmt=" ", use_line_collection=True)
plt.plot(t_col, y_pred, color = 'red') 
plt.title("Support Vector Regression, C = %f" % C)
plt.xlabel("Time in (s)")
plt.ylabel("Amplitude")
plt.show()