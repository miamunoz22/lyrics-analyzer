from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
nltk.download('stopwords')
import seaborn as sns
import pandas as pd
import numpy as np
import nltk
import os
import re

''' The three formats in Mirex data follow the same file 
naming scheme: so file 004.mp3, 004.txt and 004.mid 
correspond to the same song. There are 903 mp3 files but only
764 lyrics files in this dataset'''

# Using song-cleaner.csv because the original dataset had one row with an extra column
songs = pd.read_csv('dataset/songs-cleaner.csv', dtype=d_types)

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

corpus = list(lyrics_df['lyrics'])
labels = list(lyrics_df['cluster'])
#labels = list(lyrics_df['description'])

x_train, x_test, y_train, y_test = train_test_split(corpus, labels, test_size=0.3, random_state=22)

# Bag-of-words
bag = CountVectorizer(max_features=1000)
x_train_bag = bag.fit_transform(x_train)
x_test_bag = bag.transform(x_test)

# Initialize/train MNB
model = MultinomialNB()
model.fit(x_train_bag, y_train)

# Test the model
predictions = model.predict(x_test_bag)

# Evaluate the model
acc = accuracy_score(y_test, predictions)
print(f"Accuracy: {acc:.2f}")

# Plot results
confused = confusion_matrix(y_test, predictions)
sns.heatmap(confused, annot=True, fmt="d", cmap="Purples")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

print(classification_report(y_test, predictions))