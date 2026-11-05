import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
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


# duration * sample_rate = how many samples
y_disco.shape

disco_fft = np.fft.fft(y_disco)
disco_fft.shape
disco_fft[0]
# what we get is a complex number representation 



# we have the real number now with the associated magnitude value
magnitude_spectrum_disco = np.abs(disco_fft)
magnitude_spectrum_disco[0]


def mag_spec(signal, title, sr, f_ratio=1):
    result_fft = np.fft.fft(signal)
    mag_spec = np.abs(result_fft)
    
    
    plt.figure(figsize=(17,10))
    freq_bins = np.linspace(0, sr, len(mag_spec))
    
    num_freq_bins = int(len(freq_bins) * f_ratio)
    
    plt.bar(x= freq_bins[:num_freq_bins], height= mag_spec[:num_freq_bins])
    plt.xlabel('frequency (hz)')
    plt.title(title)
    
    plt.show()




mag_spec(y_metal, 'metal', sr, 0.5)
mag_spec(y_hiphop, 'hiphoo', sr, 0.5)
mag_spec(y_disco, 'disco', sr, 0.001)
mag_spec(y_rock, 'rock', sr, 0.5)
