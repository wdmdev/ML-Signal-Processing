#%% 
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sounddevice as sd # only needed for playing
import soundfile as sf # only needed if playing does not work
from lms import lms
from nlms import nlms
from rls import rls
from mse import mse

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# Load audio files
x, Fs = librosa.load(os.path.join(DATA_DIR, 'highwaynoise.wav'), sr=16000)
y, Fs = librosa.load(os.path.join(DATA_DIR, 'noisy_speech.wav'), sr=16000)

# one of lms, nlms, rls
adaptive_algo = 'lms'

# Filter Length
L = 30

# parameters for lms 
mu_lms = 0.5  # step size

# parameters for nlms
mu_nlms = 0.2  # normalized step-size
delta = 1e-2  # regularization parameter

# parameters for rls
beta = 0.997  # forget factor
lambda_ = 1e2  # regularization

# Switch between adaptive algorithms
if adaptive_algo == 'lms':
    yhat, _ = lms(x, y, L, mu_lms)
elif adaptive_algo == 'nlms':
    yhat, _ = nlms(x, y, L, mu_nlms, delta)
elif adaptive_algo == 'rls':
    yhat, _ = rls(x, y, L, beta, lambda_)

# Plot the spectrograms of the signals
# noisy speech
spec_y = librosa.stft(y, n_fft=512, hop_length=32, center=True)
y_db = librosa.amplitude_to_db(abs(spec_y))
plt.figure(figsize=(14, 3))
plt.title("NOISY SPEECH")
librosa.display.specshow(y_db, sr=Fs)

# noise
spec_x = librosa.stft(x, n_fft=512, hop_length=32, center=True)
x_db = librosa.amplitude_to_db(abs(spec_x))
plt.figure(figsize=(14, 3))
plt.title("HIGHWAY NOISE")
librosa.display.specshow(x_db, sr=Fs)

# filter output
spec_yhat = librosa.stft(y-yhat, n_fft=512, hop_length=32, center=True)
yhat_db = librosa.amplitude_to_db(abs(spec_yhat))
plt.figure(figsize=(14, 3))
plt.title("FILTER ERROR SIGNAL")
librosa.display.specshow(yhat_db, sr=Fs)

plt.show()

# Play filter output
# sd.play(y-yhat, Fs)
# if you cannot get the sounddevice to play, use this line to write the file to wav instead
sf.write(os.path.join(os.path.dirname(__file__), '5_4_1_filter_output.wav'), y-yhat, Fs, subtype='PCM_24')

# read original speech without noise
o, Fs = librosa.load(os.path.join(DATA_DIR, 'voice.wav'), sr=16000)

# Compute MSE, by comparing original speech to output from algorithm, so enhanced speech
e = mse(y-yhat, o)
print('Mean squared error: ', e)