# Lyrics emotion classification

In this project, I benchmarked several models to classify emotions associated with lyrics from a pre-labeled dataset of songs. 

## Data collection ##
The project uses the Multi-modal MIREX Emotion Dataset from Kaggle. The annotation strategy consisted of 3 raters assessing a given song's associated emotion, and only a subset of agreement by 2 out of 3 was extracted. The categorical classification led to 5 "clusters": Cluster 1 (passionate, rousing, confident, boisterous, rowdy), Cluster 2 (frollicking, cheerful, fun, sweet, amiable/good natured), Cluster 3 (literate, poignant, wistful, bittersweet, autumnal, brooding), Cluster 4 (humorous, silly, campy, quirky, whimsical, witty, wry), Cluster 5 (aggressive, fiery, tense/anxious, intense, volatile, visceral).

The dataset contains 903 audio clips (30-sec), 764 lyric, and 193 midis. For the purpose of this project, I only used the lyrics (764 songs).

The data can be found here: https://www.kaggle.com/datasets/imsparsh/multimodal-mirex-emotion-dataset/data

The lyrics folder contains a text file for each song that contain lyrics. The 'split-by-categories.bat' file was intended to organize the lyrics files into cluster folders, but for this project, it was used to pull the cluster label for the lyrics instead. 

After downloading the dataset from kaggle, you can clean the data by running "python3 lyrics.py"

## Lyrics EDA ##
Before attempting to classify the emotion associated with a song, we should explore the relatively small dataset. After pre-processing, the dataset consisted of 764 songs with 556 unique artists and over 475k words (no stopwords). The vocabulary size was over 7000. 

"lyrics_eda.py" gives us a sense for distributions of the clusters as well as a feel for the dataset's biases. For example, cluster 3 (literate, poignant, wistful, bittersweet, autumnal, brooding) and cluster 4 (humorous, silly, campy, quirky, whimsical, witty, wry) stand out in the dataset with around 170+ songs whereas the remaining clusters were assigned to less than 140 songs. Number of unique artists, on the other hand, were more balanced across the clusters, although slightly favoring cluster 3. We can also see that no decade besides a tight range within the 90s makes up more than 25% of the release years. The 90s music scene was heavily influenced by grunge, hip-hop/rap, pop, alternative rock, electronic dance music (EDM), and R&B and Soul. The dataset's imbalance toward cluster 3 and 4 may imply that the subject's music prefereneces favored genres such as alternative rocka and pop in the 90s. 

This file includes the function "song_blurb" that outputs the first couple verses of a song by an Artist in the dataset. The top_artists dictionary stores the top artist for each cluster. For example, Chuck Berry (in dataset as 'Berry, Chuck') is the top artist for cluster 2 (frollicking, cheerful, fun, sweet, amiable/good natured). Given the word clouds generated for each cluster, it sounds like these artists, generally, have **got** to **know** a lot about **love**...**baby**. 

## Choosing a model ##
For the task of assigning clusters based on lyrics, I decided to go with the Logistic Regression model as it had a higher score compared to Random Forest Classifier, Multinomial NB, and Linear SVC. Logistic regression is also a typical model for this use case(?) becuase it XXX. 

Without any hyperparameter tuning, the baseline Logistic Regression model had [RESULTS]. After grid searching to find improved hyperparameters, the new LR model had [RESULTS]. The less than ideal results are not entirely surprising given the very small dataset that also happened to be imbalanced in favor of Cluster 3 (which incidentally was the cluster that was classified with highest accuracy). 

## Challenges and next steps ##
There are several shortcomings of using the Mirex dataset. For one, the dataset is biased toward the music taste of one person that had a preference for songs released largely within a 50-year timeframe with an emphasis on post-1980 to early 2000s music. The dataset was also imbalanced in favor of clusters 3 and 4. Although maybe more of a personal preference, the labeling of the clusters was not very intuitive and it could be difficult to really explain why some labels like fun and boisterous didn't belong to the same cluster. Another issue that arises when trying to classify an emotion associated with a song is that, for any given song, multiple emotions could be associated with it that span more than 1 cluster, which can also vary depending on the mood or emotional state that a listener is in at the time that they listen to the song. For example, Adele's "All I Ask" may sound a lot like cluster 5 when listening to immediately after a breakup but maybe sound more like cluster 3 after time has passed.
