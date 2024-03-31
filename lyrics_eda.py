import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sb

import lyrics


''' ADD SUMMARY '''

lyrics_df = lyrics.lyrics_df
songs = lyrics.songs

fig, axes = plt.subplots(2, 1, figsize=(10, 8)) 

sb.countplot(lyrics_df['cluster'], ax=axes[0])
axes[0].set_title('Cluster Data Count & Spread')



plt.tight_layout()
plt.show()
plt.close()

fig, axes = plt.subplots(2, 1, figsize=(10, 8)) 

ax.axis('off')
table = ax.table(cellText=data, loc='center')
axes[0].set_title('Corpus stats')



plt.tight_layout()
plt.show()
plt.close()

# Insert corpus stats table: number of unique songs, unique artists, average songs per artists, total words with stopwords, average words per song, total words with no stopwords, vocabulary size with no stopwords


# table with cluster stats: number of songs in each cluster, unique artists, unique 

# Most frequent words ranks with frequncy and average frequency

# column charts with song years

# table with a few artists that were the at the top of a cluster. 
# So top three artists with the most CLuster 1 songs
# Give examples of songs with lyrics blurb
# Maybe even a soundbite or link to youtube?

# Show word cloud




