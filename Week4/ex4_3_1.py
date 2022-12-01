#%% 
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sounddevice as sd # only needed for playing
import soundfile as sf # only needed if playing does not work
from lms import lms

# Load audio files
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
y, Fs = librosa.load(os.path.join(DATA_DIR, 'voice.wav'))
x, Fs = librosa.load(os.path.join(DATA_DIR, 'noisy_speech.wav'))

# if you use notebooks, use this snippet to create player
# IPython.display.Audio('data/voice.wav')
# IPython.display.Audio('data/noisy_speech.wav')

# Filter length
L = 128

# Adaptive LMS filter
mu = 0.8
yhat = lms(x, y, L, mu)
# mu = 0.02
# delta = 1e-6
# yhat = nlms(x, y, L, mu, delta)

# Plot the spectrograms of the filtered signals
# speech
spec_x = librosa.stft(y, n_fft=512, hop_length=32, center=True)
y_db = librosa.amplitude_to_db(abs(spec_x))
plt.figure(figsize=(14, 5))
plt.title("SPEECH")
librosa.display.specshow(y_db, sr=Fs)

# noisy speech
spec_u = librosa.stft(x, n_fft=512, hop_length=32, center=True)
x_db = librosa.amplitude_to_db(abs(spec_u))
plt.figure(figsize=(14, 5))
plt.title("NOISY SPEECH")
librosa.display.specshow(x_db, sr=Fs)

# filter output
spec_dhat = librosa.stft(yhat, n_fft=512, hop_length=32, center=True)
yhat_db = librosa.amplitude_to_db(abs(spec_dhat))
plt.figure(figsize=(14, 5))
plt.title("WIENER FILTER OUTPUT")
librosa.display.specshow(yhat_db, sr=Fs)

plt.show()

# Play filter output
# sd.play(dhat, Fs)
# if you cannot get the sounddevice to play, use this line to write the file to wav instead
sf.write(os.path.join(os.path.dirname(__file__), 'filter_output.wav'), yhat, Fs, subtype='PCM_24')
