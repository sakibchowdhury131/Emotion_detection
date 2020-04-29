#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 23:23:15 2020

@author: sakib
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from read_stanford_sentiment_treebank import read_data
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
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



from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(dfNew['phrase']).toarray()
y = dfNew.iloc[:, 1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('F-measure: %.3f' % f1_score(y_test, y_pred))


