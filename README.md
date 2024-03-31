# Lyrics emotion classification

In this project, I benchmarked several models to classify emotions associated with lyrics from a pre-labeled dataset of songs. 

## Data collection ##
The project uses the Multi-modal MIREX Emotion Dataset from Kaggle. The annotation strategy consisted of 3 raters assessing a given song's associated emotion, and only a subset of agreement by 2 out of 3 was extracted. The categorical classification led to 5 "clusters": Cluster 1 (passionate, rousing, confident, boisterous, rowdy), Cluster 2 (frollicking, cheerful, fun, sweet, amiable/good natured), Cluster 3 (literate, poignant, wistful, bittersweet, autumnal, brooding), Cluster 4 (humorous, silly, campy, quirky, whimsical, witty, wry), Cluster 5 (aggressive, fiery, tense/anxious, intense, volatile, visceral).

The dataset contains 903 audio clips (30-sec), 764 lyric, and 193 midis. For the purpose of this project, I only used the lyrics (764 songs).

The data can be found here: https://www.kaggle.com/datasets/imsparsh/multimodal-mirex-emotion-dataset/data

The lyrics folder contains a text file for each song that contain lyrics. The 'split-by-categories.bat' file was intended to organize the lyrics files into cluster folders, but for this project, it was used to pull the cluster label for the lyrics instead. 

After downloading the dataset from kaggle, you can clean the data by running "python3 lyrics.py"

## Lyrics EDA ##
Before attempting to classify the emotion associated with a song, we should explore the relatively small dataset. 

Running "python3 lyrics_eda.py" gives us a sense that XXX

## Bechmarking models ##


## Challenges and next steps ##
There are several shortcomings of using the Mirex dataset. For one, the dataset is biased toward the music taste of one person that had a preference for songs released largely within a X-year timeframe. The dataset was also imbalanced in favor of XX.  Although maybe more of a personal preference, the labeling of the clusters was not very intuitive and it could be difficult to really explain why X and X didn't belong to the same cluster. Another issue that arises when trying to classify an emotion associated with a song is that, for any given song, multiple emotions could be associated with it that span more than 1 cluster, which can also vary depending on the mood or emotional state that a listener is in at the time that they listen to the song. For example, Adele's "All I Ask" may sound a lot like cluster 5 when listening to immediately after a breakup but maybe sound more like cluster 3 after time has passed.
