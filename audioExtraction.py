import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load audio file
file_path = "Taylor Swift - The Way I Loved You (Taylor's Version).mp3"
y, sr = librosa.load(file_path, sr=None)

# Create figure and subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Plot waveform
axs[0].set(title='Waveform', xlabel='Time (s)', ylabel='Amplitude')
librosa.display.waveshow(y, sr=sr, ax=axs[0])

# Plot spectrogram
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
axs[1].set(title='Spectrogram', xlabel='Time (s)', ylabel='Frequency (Hz)')
img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear', ax=axs[1])
fig.colorbar(img, ax=axs[1], format='%+2.0f dB')

# Zooming in waveform
n0 = 9000
n1 = 9100
plt.figure(figsize=(14, 5))
plt.plot(y[n0:n1])
plt.title('Zoomed-in Waveform')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.grid()

plt.tight_layout()
plt.show()
