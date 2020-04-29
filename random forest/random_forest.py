#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 12:09:29 2020

@author: sakib
"""

import pandas as pd
from read_stanford_sentiment_treebank import read_data
import re
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score



dataset = read_data('/media/sakib/alpha/work/EmotionDetectionDir/NaiveBayes/stanfordSentimentTreebank')
dataset['sentiment_values'] = pd.to_numeric(dataset['sentiment_values'], downcast = 'float')
dataset['sentiment_values'] = (dataset['sentiment_values'] >= 0.4).astype(float)

review_lines = list()
lines = dataset['Phrase'].values.tolist()
sentiment = dataset['sentiment_values']


for line in lines:
    review = re.sub('[^a-zA-Z]', ' ', line)
    review = review.lower()
    review_lines.append(review)


df = pd.DataFrame(
    {'phrase': review_lines,
     'sentiment': sentiment
     })

df['phrase'] = df['phrase'].str.lstrip()

filter = df['phrase'] != ""
dfNew = df[filter]



from sklearn.feature_extraction.text import TfidfVectorizer
# Create feature vectors
vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)
X = vectorizer.fit_transform(dfNew['phrase'])
y = dfNew['sentiment']


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('F-measure: %.3f' % f1_score(y_test, y_pred))



