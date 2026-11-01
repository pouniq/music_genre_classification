# the major feature that we use in Machine learning
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd



BASE_PATH = 'Data/genres_original/'
hiphop = 'hiphop/hiphop.00008.wav'
disco = 'disco/disco.00008.wav'
metal = 'metal/metal.00008.wav'
rock = 'rock/rock.00055.wav'



ipd.Audio(BASE_PATH+metal)
ipd.Audio(BASE_PATH+disco)
ipd.Audio(BASE_PATH+hiphop)
ipd.Audio(BASE_PATH+rock)

y_disco, sr = librosa.load('Data/genres_original/disco/disco.00007.wav', sr = 22050, duration = 3)
y_metal, _ = librosa.load('Data/genres_original/metal/metal.00007.wav', sr = 22050)
y_hiphop, _ = librosa.load('Data/genres_original/hiphop/hiphop.00007.wav', sr = 22050)
y_rock, _ = librosa.load('Data/genres_original/rock/rock.00007.wav', sr = 22050)


FRAME_SIZE = 2048
HOP_SIZE = 512


s_disco = librosa.stft(y_disco, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
s_disco[0][0]
s_disco.shape


# we are taking the magnitude
Y_disco = np.abs(s_disco) ** 2
Y_disco.shape

def plot_spectogram(Y, sr, hop_length, y_axis = 'linear'):
    plt.figure(figsize=(17,10))
    librosa.display.specshow(Y, sr=sr, hop_length=hop_length, x_axis='time', y_axis=y_axis)
    plt.colorbar(format='%+2.f')
    
plot_spectogram(Y_disco, sr, HOP_SIZE)

# how we actually hear sound is Logrithmic and not this linear 
# and this is not a good representation of how we hear things


## log amplitude Spectogram 
Y_disco_db = librosa.power_to_db(Y_disco)
plot_spectogram(Y_disco_db, sr, HOP_SIZE)


# log-frequecy spectogram
plot_spectogram(Y_disco_db, sr, HOP_SIZE, y_axis='log')
plot_spectogram(Y_disco_db, sr, HOP_SIZE, y_axis='log')
plot_spectogram(Y_disco_db, sr, HOP_SIZE, y_axis='log')


## Making Spectogram function 

def make_spec(signal, FRAME_SIZE, HOP_SIZE, sr=sr):
    stft_audio = librosa.power_to_db(np.abs(librosa.stft(signal, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)) ** 2)
    plot = plot_spectogram(stft_audio, sr, HOP_SIZE, y_axis='log')
    return stft_audio, plot
    
make_spec(y_metal, FRAME_SIZE=FRAME_SIZE, HOP_SIZE= HOP_SIZE, sr=sr)