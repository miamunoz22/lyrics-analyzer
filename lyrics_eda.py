import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sb

import lyrics
from prettytable import PrettyTable


''' After some preliminary cleaning of the songs and the lyrics data, this file
explores the data further to get any useful insights ahead of modeling. '''

lyrics_df = lyrics.lyrics_df
songs = lyrics.songs
merged = pd.merge(lyrics_df.iloc[:,1:], songs, on='track_id', how='inner')

# Extracting summary data
merged.head()
merged.reset_index(inplace=True)
merged.info()
merged['track_id'] = merged['track_id'].astype('string')
merged.duplicated('track_id').any() # Check for any duplicate tracks


# Exploring counts and distributions
fig, axes = plt.subplots(2, 1, figsize=(15, 8)) 

# Cluster count
sb.countplot(lyrics_df['cluster'], ax=axes[0])
axes[0].set_title('Cluster Data Count & Spread')

# Songs by year in the original dataset
sb.countplot(songs['Year'], ax=axes[1])
axes[1].set_title('Songs by Year')

plt.setp(axes[1].get_xticklabels(), rotation=90) 
plt.tight_layout()
plt.show()
plt.close()

plt.figure(figsize=(10,6)) 
sb.boxplot(x='Year', data=songs)
plt.title('Songs by Year - boxplot')
plt.show()
plt.close()



tracks = len(merged['track_id'])
uni_artists = len(set(merged['Artist']))
avg_per_artists =round(tracks / uni_artists,0)
corpus = list(merged['lyrics'])
avg_words_per_song = round(np.mean([len(x) for x in corpus]),0)
corpus_size = round(np.sum([len(x) for x in corpus]),0)
vocabulary = set(' '.join(corpus).split())
vocab_size = len(vocabulary)

# Table with corpus stats
corpus_stats = PrettyTable()
corpus_stats.field_names = ["Statistic", "Value"]
corpus_stats.add_row(["Number of songs", tracks])
corpus_stats.add_row(["Number of artists", uni_artists])
corpus_stats.add_row(["Average songs per artist", avg_per_artists])
corpus_stats.add_row(["Total words (no stopwords)", corpus_size])
corpus_stats.add_row(["Average words per song", avg_words_per_song])
corpus_stats.add_row(["Vocabulary size (no stopwords)", vocab_size])

print(corpus_stats)

# Table with cluster stats: number of songs in each cluster, unique artists, unique 
artists_gb = merged.groupby('cluster')['Artist']
tracks_gb = merged.groupby('cluster')['track_id']

artists_gb.nunique() # Number of artists in each cluster
tracks_gb.nunique() # Number of songs in each cluster

# table with a few artists that were the at the top of a cluster. 
# So top three artists with the most CLuster 1 songs
# Give examples of songs with lyrics blurb
# Maybe even a soundbite or link to youtube?


# Most frequent words ranks with frequncy and average frequency
# Show word cloud, maybe one for each cluster