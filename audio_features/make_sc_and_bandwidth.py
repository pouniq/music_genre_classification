import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd

BASE_PATH = '../Data/genres_original/'
hiphop = 'hiphop/hiphop.00008.wav'
disco = 'disco/disco.00008.wav'
metal = 'metal/metal.00008.wav'
rock = 'rock/rock.00055.wav'



ipd.Audio(BASE_PATH+metal)
ipd.Audio(BASE_PATH+disco)
ipd.Audio(BASE_PATH+hiphop)
ipd.Audio(BASE_PATH+rock)

y_disco, sr = librosa.load('../Data/genres_original/disco/disco.00007.wav', sr = 22050, duration = 3)
y_metal, _ = librosa.load('../Data/genres_original/metal/metal.00007.wav', sr = 22050, duration=3)
y_hiphop, _ = librosa.load(BASE_PATH+hiphop, sr = 22050, duration=3)

FRAME_SIZE = 2048
HOP_SIZE = 512



sc_disco = librosa.feature.spectral_centroid(y = y_disco, sr = sr, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)[0]
sc_metal = librosa.feature.spectral_centroid(y = y_metal, sr = sr, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)[0]
sc_hiphop = librosa.feature.spectral_centroid(y = y_hiphop, sr = sr, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)[0]


sc_hiphop.shape

## visualize sc
frames = range(len(sc_disco))
t = librosa.frames_to_time(frames)

plt.figure(figsize=(25,10))
plt.plot(t, sc_disco, color = 'red')
plt.plot(t, sc_metal, color = 'blue')
plt.plot(t, sc_hiphop, color = 'purple')
plt.show()




sb_disco = librosa.feature.spectral_bandwidth(y = y_disco, sr = sr, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)[0]
sb_metal = librosa.feature.spectral_bandwidth(y = y_metal, sr = sr, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)[0]
sb_hiphop = librosa.feature.spectral_bandwidth(y = y_hiphop, sr = sr, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)[0]

frames = range(len(sb_disco))
t = librosa.frames_to_time(frames)

plt.figure(figsize=(25,10))
plt.plot(t, sb_disco, color = 'red')
plt.plot(t, sb_metal, color = 'blue')
plt.plot(t, sb_hiphop, color = 'purple')
plt.show()