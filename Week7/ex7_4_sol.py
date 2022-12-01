#%%
import numpy as np
import matplotlib.pyplot as plt
import math
from ex7_3_2_sol import DFT

def STFT(x, frame_size, hop_size):
    n_frames = math.floor((len(x)-frame_size)/hop_size)+1 # calculating number of frames
    n_bins = frame_size//2 + 1 # calculating number of frequencies   
    
    X = np.zeros((n_bins, n_frames), dtype=complex)

    window = np.hanning(frame_size)

    for i in range(n_frames): 
        # Hint: prepare a block of signal for the fft
        frame = x[i*hop_size:i*hop_size+frame_size]*window  # solution  

        # Hint: implement equation 2.62, multiply signal with the window 
        X[:, i] = DFT(frame)[:frame_size//2+1] # solution  

    return X

#%%
tend = 1
N = 1024
Ts = tend/N
Fs = 1/Ts
t = np.arange(N)*Ts;
freq = 10
freq2 = 100
s = np.sin(2*np.pi*freq*t)
s2 = np.sin(2*np.pi*freq2*t)

# Generate a combination of s and s2
comb = np.zeros(N)
comb[:N//2] = s[:N//2]
comb[N//2:] = s2[N//2:]
comb2 = (s+s2)/2

# Compute DFT of the signals
s_dft = DFT(comb)
s_dft2 = DFT(comb2)
freqs = np.arange(N)*Fs/N

# Use one-sided DFTs
s_dft = s_dft[:N//2+1]
s_dft2 = s_dft2[:N//2+1]
freqs = freqs[:N//2+1]

# Plot DFTs
for comb_i, s_dft_i in zip([comb, comb2], [s_dft, s_dft2]):
    plt.figure(figsize=(14, 3))
    plt.subplot(1,2,1)
    plt.plot(t, comb_i)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.subplot(1,2,2)
    plt.stem(freqs, abs(s_dft_i), 'r')
    plt.xlabel('Frequency')
    plt.ylabel('|X(freq)|')
    plt.grid()
    plt.tight_layout()

#%%
frame_size = 256
hop_size = 64

# Compute spectrograms
s_stft = STFT(comb, frame_size, hop_size);
s_stft2 = STFT(comb2, frame_size, hop_size);

# Create time frequency axes
t = np.arange(s_stft.shape[1])*Ts*frame_size
f = np.arange(frame_size)*Fs/frame_size;
f = f[:frame_size//2+1]

# Plot spectrograms
plt.figure(figsize=(14, 6))
for i, s_stft_i in enumerate([s_stft, s_stft2]):
    plt.subplot(2,1,i+1)
    plt.pcolormesh(t, f, 20*np.log10(abs(s_stft_i)), shading='auto')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar()
plt.tight_layout()
plt.show()