import librosa 
import matplotlib.pyplot as plt
import IPython.display as ipd


BASE_PATH = 'Data/genres_original/'
metal = 'metal/metal.00008.wav'

mel_metal , sr = librosa.load(BASE_PATH+metal)


## mel filter banks

filter_banks = librosa.filters.mel(n_fft=2048, sr=22050, n_mels=20)
filter_banks.shape


plt.figure(figsize=(17,10))
librosa.display.specshow(filter_banks, sr=sr, x_axis='linear')
plt.show()

## to create mel-spectrograms you need to go through lots of process
### first you need to create the vanilla spectrograms then create the 
### fitler banks then do matrix multiplcation to both of them

mel_spec = librosa.feature.melspectrogram(y=mel_metal, sr=sr, n_fft=2048, hop_length=512, n_mels = 128)



mel_log_metal = librosa.power_to_db(mel_spec)
plt.figure(figsize=(17,10))
librosa.display.specshow(mel_log_metal, sr=sr, x_axis='time', y_axis='mel')
plt.show()