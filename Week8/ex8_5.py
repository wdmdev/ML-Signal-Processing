#%%
import os
import librosa
import librosa.display
import sounddevice as sd
import soundfile as sf # only needed if playing does not work
import numpy as np
import matplotlib.pyplot as plt
from ICA import ICA

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# load data
x, Fs = librosa.load(os.path.join(DATA_DIR, 'mixture_instantaneous.wav'), mono=False, sr = 44100) 
x = x.T

# run ICA
mu = 0.1
iters = 200
num_components = 2

W = ICA(x, mu, num_components, iters, 'superGauss')

# Normalize unmixing matrix
W = np.divide(W, np.max(W))

# Compute unmixed signals
y = (W @ x.T).T

# Play the first component
print('Play ICA estimated source 1')
# sd.play(y[:,0], Fs)
sf.write(os.path.join(os.path.dirname(__file__), 'ica_source_1.wav'), y[:,0], Fs, subtype='PCM_24')

# Play the second component
print('Play ICA estimated source 2')
# sd.play(y[:,1], Fs)
sf.write(os.path.join(os.path.dirname(__file__), 'ica_source_2.wav'), y[:,1], Fs, subtype='PCM_24')

# Plot the spectrograms of the filtered signals
# mix
spec_x = librosa.stft(librosa.to_mono(np.transpose(x)), n_fft=512, hop_length=32, center=True)
x_db = librosa.amplitude_to_db(abs(spec_x))
plt.figure(figsize=(14, 5))
plt.title("MIX")
librosa.display.specshow(x_db, sr=Fs)

# source 1
spec_s1 = librosa.stft(y[:,0], n_fft=512, hop_length=32, center=True)
s1_db = librosa.amplitude_to_db(abs(spec_s1))
plt.figure(figsize=(14, 5))
plt.title("SOURCE 1")
librosa.display.specshow(s1_db, sr=Fs)

# source 2
spec_s2 = librosa.stft(y[:,1], n_fft=512, hop_length=32, center=True)
s2_db = librosa.amplitude_to_db(abs(spec_s2))
plt.figure(figsize=(14, 5))
plt.title("SOURCE 2")
librosa.display.specshow(s2_db, sr=Fs)