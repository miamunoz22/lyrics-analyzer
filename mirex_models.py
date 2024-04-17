from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
import os

import prep

'''ADD SUMMARY'''

lyrics_df = prep.lyrics_df

corpus = list(lyrics_df['lyrics'])
labels = list(lyrics_df['cluster'])
cluster_ext  = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5']

# tf-idf matrix
vec = TfidfVectorizer(max_features = 2000, sublinear_tf=True, norm='l2', encoding='latin-1', ngram_range=(1,2))
features = vec.fit_transform(corpus).toarray()

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=22)

# Looping through the various models of interest
models = [MultinomialNB(), LinearSVC(), LogisticRegression(random_state=0), RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0)]

CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))

entries = []

for model in models:
    model_name = model.__class__.__name__
    accs = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
    for fold, accuracy in enumerate(accs):
        entries.append((model_name, fold, accuracy))
    
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold', 'accuracy'])

sb.boxplot(x='model_name', y='accuracy', data=cv_df)
sb.stripplot(x='model_name', y='accuracy', data=cv_df, size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()

# Choosing Logistic Regression model as the model to move forward with
params = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2'], 'solver': ['liblinear', 'lbfgs'], 'class_weight': [None, 'balanced', {0: 1, 1: 5}], 'tol': [1e-4, 1e-3, 1e-2]}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(models[2], params, cv=5, scoring='accuracy')
grid_search.fit(x_train, y_train)

# Get the best hyperparameters
best_c = grid_search.best_params_['C']
best_penalty = grid_search.best_params_['penalty']
best_solver = grid_search.best_params_['solver']
best_weight = grid_search.best_params_['class_weight']
best_tol = grid_search.best_params_['tol']

# Train the LogReg model with the best hyperparameters
best_log_classifier = LogisticRegression(C=best_c, penalty=best_penalty, solver=best_solver, class_weight=best_weight, tol=best_tol)
best_log_classifier.fit(x_train, y_train)

# Evaluate the model on the test set
y_pred = best_log_classifier.predict(x_test)

# Plot results
confused = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sb.heatmap(confused, annot=True, fmt='d', xticklabels=cluster_ext, yticklabels=cluster_ext, cmap='Purples')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

print(classification_report(y_test, y_pred))

# Achieved a whopping accuracy of 38%!

# INSERT COLUMN CHART WITH THE RESULTS'''