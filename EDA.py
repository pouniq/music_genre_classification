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






