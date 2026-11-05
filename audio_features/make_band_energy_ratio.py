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


FRAME_SIZE = 2048
HOP_SIZE = 512


s_disco = librosa.stft(y_disco, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
s_disco_transpose = s_disco.T


s_metal = librosa.stft(y_metal, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)


# band energy ratio
def calculate_split_frequency(spectrogram, split_freq, sr):
    freq_range = sr / 2
    freq_delta_per_bin = freq_range / spectrogram.shape[0]
    split_freq_bin = np.floor(split_freq / freq_delta_per_bin)
    return int(split_freq_bin)


calculate_split_frequency(s_disco, 2000, 22050)


def calculate_band_energy_ratio(spectrogram,split_freq, sr):
    split_freq_bin = calculate_split_frequency(spectrogram, split_freq, sr)
    
    # move to the power spectrogram
    power_spec = np.abs(spectrogram) ** 2
    power_spec = power_spec.T
    
    band_energy_ratio = []
    for freq_frame in power_spec:
        low_freq_band = np.sum(freq_frame[:split_freq_bin])
        high_freq_band = np.sum(freq_frame[split_freq_bin:])
        ber_current_frame = low_freq_band / high_freq_band 
        band_energy_ratio.append(ber_current_frame)
        
    return np.array(band_energy_ratio)

    
ber_disco = calculate_band_energy_ratio(s_disco, 2000, 220500) 
ber_metal = calculate_band_energy_ratio(s_metal, 2000, 220500) 
ber_disco.shape


# band Energy Ratio
frames = range(len(ber_disco))
t = librosa.frames_to_time(frames=frames)

plt.figure(figsize=(25,10))
plt.plot(t, ber_disco, color= 'red')
plt.plot(t, ber_metal, color= 'blue')
plt.show()


if __name__ == "__main__":
    print('ber loaded')