import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
import pandas as pd
import numpy as np
import os
import re

''' The three formats in Mirex data follow the same file 
naming scheme: so file 004.mp3, 004.txt and 004.mid 
correspond to the same song. There are 903 mp3 files but only
764 lyrics files in this dataset'''

# Using song-cleaner.csv because the original dataset had one row with an extra column
songs = pd.read_csv('dataset/songs-cleaner.csv')

# Inspecting the data
songs.head
songs.shape
songs.isnull().sum()
songs[songs.isnull().any(axis=1)] # Shows 6 rows with null values

# Cleaning the data
songs.dropna(inplace=True) # Leaves us with 897 entries
songs['Year'] = songs['Year'].astype(int)
songs['Last Modified'] = pd.to_datetime(songs['Last Modified'], format='%d-%m-%Y')

# Re-assign the volues in filename column to drop .mp3 and change header to ID
songs['Filename'] = songs['Filename'].str.replace('.mp3', '')
songs.rename(columns={'Filename': 'track_id'}, inplace=True)

# Creating a simplified dataframe
lyrics_df = pd.DataFrame(songs['track_id'])
lyrics_df['cluster'] = None
lyrics_df['description'] = None
lyrics_df['lyrics'] = None


# importing the bat file that has info on clusters for each song
with open('dataset/split-by-categories-lyrics.bat', 'r') as bat_file:
    for line in bat_file:
        if re.search(r'\bmove\b', line):
            file = re.search(r'(\d+)', line).group()
            cluster = re.search(r'Cluster \s*(\d+)', line).group(1)
            description = re.search(r'\\(.*?)\\', line).group(1)
            lyrics_df.loc[songs['track_id'] == file, 'cluster'] = cluster
            lyrics_df.loc[songs['track_id'] == file, 'description'] = description

# helper function to open and read lyrics txt file
def get_lyrics(track_id):
    path = 'dataset/Lyrics/' + track_id + '.txt'
    try:
        with open(path, 'r') as file:
            lyrics = file.read()
            assert type(lyrics) == str
            assert lyrics is not None
            return lyrics
    except:
        return None
                
# helper function to pre-process lyrics: remove stop words, get word roots, etc.
def clean_lyrics(lyrics):
    try:
        tokens = re.findall(r'\b[^\W\d_]+\b', lyrics)
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word.lower()) for word in tokens if word.lower() not in stopwords.words('english')]
        return ' '.join(tokens)
    except:
        return None
              
            
# Adding lyrics to the lyrics_df and cleaning them within the dataframe
lyrics_df['lyrics'] = lyrics_df['track_id'].apply(get_lyrics)
lyrics_df['lyrics'] = lyrics_df['lyrics'].apply(clean_lyrics)

# Shows 133 songs with no lyrics in dataset - matches what mirex data gives
lyrics_df[lyrics_df.isnull().any(axis=1)]
lyrics_df.dropna(inplace=True)