from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential

import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
import os

import prep

'''Using a NN model separate from the bechmarked models'''

lyrics_df = prep.lyrics_df

corpus = list(lyrics_df['lyrics'])
labels = list(lyrics_df['cluster'])

x_train, x_test, y_train, y_test = train_test_split(corpus, labels, test_size=0.3, random_state=22)

# tf-idf matrix
vec = TfidfVectorizer(max_features = 2000, sublinear_tf=True, norm='l2', encoding='latin-1', ngram_range=(1,2))
train_matrix = vec.fit_transform(corpus).toarray()
test_matrix = vec.transform(x_test)


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
