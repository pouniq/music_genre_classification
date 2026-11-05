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

y_disco.shape


# extract MFCC
mfccs = librosa.feature.mfcc(y=y_disco, n_mfcc=13, sr=sr )
mfccs.shape

# visualize mfcc
plt.figure(figsize=(17,10))
librosa.display.specshow(mfccs, x_axis='time', sr=sr)
plt.colorbar(format='%+2f')
plt.show()


## derivaties of the MFCC
### delta and delta2 MFCCs
### with derivatives of MFCCs we get how MFCCs change 
### over time
delta_mfcc = librosa.feature.delta(mfccs)
delta2_mfcc = librosa.feature.delta(mfccs, order=2)

delta2_mfcc.shape

plt.figure(figsize=(17,10))
librosa.display.specshow(delta_mfcc, x_axis='time', sr=sr)
plt.colorbar(format='%+2f')
plt.show()


plt.figure(figsize=(17,10))
librosa.display.specshow(delta2_mfcc, x_axis='time', sr=sr)
plt.colorbar(format='%+2f')
plt.show()


comprensive_mfcc = np.concatenate([mfccs, delta_mfcc, delta2_mfcc])
comprensive_mfcc.shape
