#%%
import os
import random

import soundfile as sf
import scipy.signal
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sn


#%%
# load the audio data that will be used to perform music genre classification
genres = ['classical', 'electronic', 'jazz', 'metal']
basedir = os.path.join(os.path.dirname(__file__), 'data')
audios = []
labels = []
for genre in genres:
    genreDir = os.path.join(basedir, genre)
    for filename in os.listdir(genreDir):
        filepath = os.path.join(genreDir, filename)
        audio, fs = sf.read(filepath)
        audios.append(audio)
        labels.append(genre)
#%%
# stft parameters
frame_size = 516
hop_size = 256
window = 'hanning'

# select a random audio file
random_audio = random.choice(audios)

# plot
fig, axes = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
t = np.arange(len(random_audio))/fs
axes[0].plot(t, random_audio)
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitude')
overlap_size = frame_size - hop_size
f, t, S = scipy.signal.stft(random_audio, fs=fs, nperseg=frame_size, noverlap=overlap_size)
S_dB = 20*np.log10(abs(S))
axes[1].pcolormesh(t, f, S_dB, shading='auto')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Frequency (Hz)')
fig.tight_layout()
plt.show()

#%%
n = len(audios)
features = np.empty((n, 2))

for i, audio in enumerate(audios):
    _, _, S = scipy.signal.stft(audio, nperseg=frame_size, noverlap=overlap_size)
    S = abs(S)
    
    # spectral centroid
    n_bins = len(S);
    bins = np.arange(n_bins).reshape(-1, 1)
    spectral_centroid = np.sum(bins*S, axis=0)/np.sum(S, axis=0)

    # spectral rolloff
    cum_energy = np.cumsum(S, axis=0);
    cum_energy_percent = cum_energy/np.sum(S, axis=0);
    spectral_rolloff = np.argmax(cum_energy_percent > 0.85, axis=0)
    
    # add to feature matrix
    features[i, 0] = spectral_centroid.mean()
    features[i, 1] = spectral_rolloff.mean()

#%%
# normalize features
mean = features.mean(axis=0)
std = features.std(axis=0)
features = (features - mean)/std

# number of nearest neighbors
k = 5

# metric
metric = 'euclidean'

# main loop
predictions = []
for i in range(n):
    # training data: all except sample i
    X_train = np.delete(features, i, axis=0)
    Y_train = np.delete(labels, i, axis=0)
    # test data: sample i
    X_test = features[i].reshape(1, -1)
    # classify
    model = KNeighborsClassifier(n_neighbors=k, metric=metric)
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    predictions.append(prediction)

accuracy = sklearn.metrics.accuracy_score(labels, predictions)
print(f'Percentage of correct classifications: {round(accuracy*100)}%')

# confusion matrix
conf_mat = sklearn.metrics.confusion_matrix(labels, predictions, normalize='true')

# plot
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sn.heatmap(conf_mat, annot=True, cmap='rocket_r', xticklabels=genres, yticklabels=genres)
plt.show()