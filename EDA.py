import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa

df_5 = pd.read_csv('gtzan_5s_features.csv')
df_5.columns



plt.subplots()

df_5.groupby('genre')['chroma2_mean'].median()




genres = df_5['genre'].unique()
n = len(genres)
ncols = 3
nrows = -(-n // ncols)  # ceil division

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                         sharex=True, sharey=True)
axes = axes.flatten()

for ax, (genre, g) in zip(axes, df_5.groupby('genre')):
    ax.plot(g['mfcc20_mean'].values)
    ax.set_title(genre)

# hide unused axes if the grid is larger than the number of genres
for ax in axes[n:]:
    ax.set_visible(False)

fig.supylabel('chroma2_mean')
fig.tight_layout()
plt.show()

sns.scatterplot(data=df_5.reset_index(), x='index', y='chroma2_mean', hue='genre', s=10)
plt.show()


box_to_check = ['chroma2_mean', 'mfcc1_mean', 'mfcc2_mean', 'mfcc6_mean']
for col in box_to_check:
    sns.boxplot(data=df_5, x='genre', y=col)
    plt.xticks(rotation=45)
    plt.show()




df_5.columns

## Chroma Features




### chroma mean

chroma_mean = df_5.iloc[:,range(4,28,2)]
chroma_mean['genre'] = df_5['genre'] 



for col in chroma_mean.drop('genre', axis=1):
    sns.boxplot(data=chroma_mean, x='genre', y=col)
    plt.title(f'{col}')
    plt.xticks(rotation=45)
    plt.show()


### chroma var
chroma_var = df_5.iloc[:,range(5,28,2)]
chroma_var['genre'] = df_5['genre']

for col in chroma_var.drop('genre', axis=1):
    sns.boxplot(data=chroma_var, x='genre', y=col)
    plt.title(f'{col}')
    plt.xticks(rotation=45)
    plt.show()


## MFCC features

mfcc_mean = df_5.iloc[:,range(52,92,2)]
mfcc_mean['genre'] = df_5['genre'] 



for col in mfcc_mean.drop('genre', axis=1):
    sns.boxplot(data=mfcc_mean, x='genre', y=col)
    plt.title(f'{col}')
    plt.xticks(rotation=45)
    plt.show()


mfcc_mean_g = mfcc_mean.groupby('genre').mean()
     
plt.figure(figsize=(12, 6))

for genre in mfcc_mean_g.index:
    plt.plot(mfcc_mean_g.columns, mfcc_mean_g.loc[genre], marker='o', label=genre)

plt.xticks(rotation=45)
plt.xlabel('coefficient')
plt.ylabel('mean value')
plt.title('Mean MFCC profile per genre')
plt.legend(title='Genre', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()
## RMS features

rms_feat_mean = df_5['rms1_mean']
rms_feat_var = df_5['rms1_var']

plt.figure(figsize=(17,10))
sns.boxplot(data=df_5, x='genre', y=rms_feat_mean)
plt.tight_layout()
plt.show()


plt.figure(figsize=(17,10))
sns.boxplot(data=df_5, x='genre', y=rms_feat_var)
plt.tight_layout()
plt.show()


##  centroid
sc_mean = df_5['centroid1_mean']

plt.figure(figsize=(17,10))
sns.boxplot(data=df_5, x='genre', y=sc_mean)
plt.tight_layout()
plt.show()



## spectral bandwidth
sb_mean = df_5['bandwidth1_mean']

plt.figure(figsize=(17,10))
sns.boxplot(data=df_5, x='genre', y=sb_mean)
plt.tight_layout()
plt.show()




## rolloff mean

plt.figure(figsize=(17,10))
sns.boxplot(data=df_5, x='genre', y='rolloff1_mean')
plt.tight_layout()
plt.show()

## zero crossing rate
plt.figure(figsize=(17,10))
sns.boxplot(data=df_5, x='genre', y='zcr1_mean')
plt.tight_layout()
plt.show()


## contrast

contrast_mean = df_5.iloc[:,range(38,48,2)]
contrast_mean['genre'] = df_5['genre'] 

for col in contrast_mean.drop('genre', axis=1):
    sns.boxplot(data=contrast_mean, x='genre', y=col)
    plt.title(f'{col}')
    plt.xticks(rotation=45)
    plt.show()


## ber mean









