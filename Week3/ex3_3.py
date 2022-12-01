#%% 
import os
import librosa
import librosa.display
import numpy as np
from scipy.linalg import toeplitz
from scipy.signal import wiener
from scipy import signal
import matplotlib.pyplot as plt
import sounddevice as sd # only needed for playing
import soundfile as sf # only needed if playing does not work
from xcorr import xcorr

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# Load audio files
d, Fs = librosa.load(os.path.join(DATA_DIR,'voice.wav'))
u, Fs = librosa.load(os.path.join(DATA_DIR, 'noisy_speech.wav'))

# if you use notebooks, use this snippet to create player
# IPython.display.Audio('data/voice.wav')
# IPython.display.Audio('data/noisy_speech.wav')

# Filter length
L = 100

# Compute Wiener filter
r_uu = xcorr(u, u, L-1)
R_uu = toeplitz(r_uu[L-1:])
r_du = xcorr(d, u, L-1)
theta = np.linalg.solve(R_uu, r_du[L-1:])

# Filter noisy signal
dhat = signal.lfilter(theta, 1, u) 

# Plot the spectrograms of the filtered signals

# speech
spec_x = librosa.stft(d, n_fft=512, hop_length=32, center=True)
d_db = librosa.amplitude_to_db(abs(spec_x))
plt.figure(figsize=(14, 5))
plt.title("SPEECH")
librosa.display.specshow(d_db, sr=Fs)

# noisy speech
spec_u = librosa.stft(u, n_fft=512, hop_length=32, center=True)
u_db = librosa.amplitude_to_db(abs(spec_u))
plt.figure(figsize=(14, 5))
plt.title("NOISY SPEECH")
librosa.display.specshow(u_db, sr=Fs)

# filter output
spec_dhat = librosa.stft(dhat, n_fft=512, hop_length=32, center=True)
dhat_db = librosa.amplitude_to_db(abs(spec_dhat))
plt.figure(figsize=(14, 5))
plt.title("WIENER FILTER OUTPUT")
librosa.display.specshow(dhat_db, sr=Fs)

# Play filter output
# sd.play(dhat, Fs)
# if you cannot get the sounddevice to play, use this line to write the file to wav instead
sf.write(os.path.join(os.path.dirname(__file__), 'filter_output.wav'), dhat, Fs, subtype='PCM_24')
