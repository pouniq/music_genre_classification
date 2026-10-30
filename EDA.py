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



for genre, g in df_5.groupby('genre'):
    plt.plot(g['chroma2_mean'].values, label=genre)
plt.legend()
plt.ylabel('chroma2_mean')
plt.show()



sns.scatterplot(data=df_5.reset_index(), x='index', y='chroma2_mean', hue='genre', s=10)
plt.show()


sns.boxplot(data=df_5, x='genre', y='chroma2_mean')
plt.xticks(rotation=45)
plt.show()



