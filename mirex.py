import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
songs.dropna(inplace=True)
songs['Year'] = songs['Year'].astype(int)
songs['Last Modified'] = pd.to_datetime(songs['Last Modified'], format='%d-%m-%Y')

# Re-assign the volues in filename column to drop .mp3 and change header to ID
songs['Filename'] = songs['Filename'].str.replace('.mp3', '')
songs.rename(columns={'Filename': 'ID'}, inplace=True)
songs['Cluster'] = None
songs['Description'] = None
songs['Lyrics'] = None

# importing the bat file that has info on clusters for each song
with open('dataset/split-by-categories-lyrics.bat', 'r') as bat_file:
    for line in bat_file:
        if re.search(r'\bmove\b', line):
            file = re.search(r'(\d+)', line, re.IGNORECASE).group()
            cluster = re.search(r'Cluster \s*(\d+)', line, re.IGNORECASE).group(1)
            description = re.search(r'\\(.*?)\\', line, re.IGNORECASE).group(1)
            songs.loc[songs['ID'] == file, 'Cluster'] = cluster
            songs.loc[songs['ID'] == file, 'Description'] = description
            
plt.hist(songs['Year'], bins=20, edgecolor='black')