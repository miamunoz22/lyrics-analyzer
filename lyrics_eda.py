import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sb
import prep
from prettytable import PrettyTable
from wordcloud import WordCloud
import mirex_models


''' After some preliminary cleaning of the songs and the lyrics data, this file
explores the data further to get any useful insights ahead of modeling. '''

lyrics_df = prep.lyrics_df
songs = prep.songs
merged = pd.merge(lyrics_df.iloc[:,1:], songs, on='track_id', how='inner')
descriptions = {'1': 'Passionate, rousing, confident, boisterous, rowdy', 
                '2': 'Rollicking, cheerful, fun, sweet, amiable/good natured', 
                '3': 'Literate, poignant, wistful, bittersweet, autumnal, brooding', 
                '4': 'Humorous, silly, campy, quirky, whimsical, witty, wry',
                '5': 'Aggressive, fiery, tense/anxious, intense, volatile, visceral'}

# Extracting summary data
merged.head()
merged.info()
merged['track_id'] = merged['track_id'].astype('string')
merged.duplicated('track_id').any() # Check for any duplicate tracks


# Exploring distributions
fig, axes = plt.subplots(2, 1, figsize=(15, 12)) 

sb.countplot(lyrics_df['cluster'], ax=axes[0]) # Cluster count
axes[0].set_title('Cluster Data Count & Spread')

sb.boxplot(x='Year', data=songs) # Songs by year distribution
plt.title('Songs by Year - boxplot')
plt.show()
plt.close()

# Table with corpus stats
tracks = len(merged['track_id'])
uni_artists = len(set(merged['Artist']))
avg_per_artists =round(tracks / uni_artists,0)
corpus = list(merged['lyrics'])
avg_words_per_song = round(np.mean([len(x) for x in corpus]),0)
corpus_size = round(np.sum([len(x) for x in corpus]),0)
vocabulary = set(' '.join(corpus).split())
vocab_size = len(vocabulary)

corpus_stats = PrettyTable()
corpus_stats.field_names = ["Statistic", "Value"]
corpus_stats.add_row(["Number of songs", tracks])
corpus_stats.add_row(["Number of artists", uni_artists])
corpus_stats.add_row(["Average songs per artist", avg_per_artists])
corpus_stats.add_row(["Total words (no stopwords)", corpus_size])
corpus_stats.add_row(["Average words per song", avg_words_per_song])
corpus_stats.add_row(["Vocabulary size (no stopwords)", vocab_size])

print(corpus_stats)

# Cluster stats number of songs in each cluster, unique artists 
artists_gb = merged.groupby('cluster')['Artist']
tracks_gb = merged.groupby('cluster')['track_id']
artists_gb.nunique() # Number of artists in each cluster
tracks_gb.nunique() # Number of songs in each cluster


top_artists = merged.groupby('cluster')['Artist'].agg(lambda x: x.mode().iloc[0])
top_artists = top_artists.to_dict()

def song_blurb(artist):
    track = merged[merged['Artist'] == artist]['track_id'].iloc[0] # first result
    track_name = merged[merged['track_id'] == track]['Title'].iloc[0]
    text = prep.get_lyrics(track)
    blurb = " ".join(text.split('\n\n')[0:2]) # first two verses
    result = "Here are the first couple verses of {} by {}: \n {}".format(track_name, artist, blurb)
    return print(result)

song_blurb(top_artists['2'])

# Most frequent words ranks with frequncy and average frequency
# Show word cloud, maybe one for each clusterfrom wordcloud 


# Generate general wordcloud
corpus = ' '.join(mirex_models.corpus) # Join all lyrics into a single string
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(corpus)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

grouped = merged.groupby('cluster')

# Create a word cloud for each cluster and display them using subplots
fig, axes = plt.subplots(5, 1, figsize=(15, 10))
for (cluster, data), ax in zip(grouped, axes.flatten()):
    # Join all lyrics for the current cluster into a single string
    all_lyrics = ' '.join(data['lyrics'])
    
    # Generate the word cloud for the current cluster
    wordcloud = WordCloud(width=400, height=200, background_color='white').generate(all_lyrics)
    
    # Display the word cloud for the current cluster
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(f'Cluster {cluster}: {descriptions[cluster]}')
    ax.axis('off')

plt.tight_layout()
plt.show()
