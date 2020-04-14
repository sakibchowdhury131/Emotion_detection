

# -*- coding: utf-8 -*-

# this model is overfitted. Training set has 99.35% accuracy. Test set fails almost completely


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('train_data.csv')

import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 30000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['content'][i])
    #review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X_train = cv.fit_transform(corpus).toarray()
y_train = dataset.iloc[:, 0].values


keywords = []

for i in range (0,30000):
    if not(y_train[i] in keywords):
        keywords.append(y_train[i])

y_train = cv.fit_transform(dataset['sentiment']).toarray()

# from sklearn.naive_bayes import GaussianNB
# classifier = GaussianNB()
# classifier.fit(X_train, y_train)
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()


classifier.add(Dense(output_dim = 128, init = 'uniform', activation = 'relu', input_dim = 1500))
classifier.add(Dense(output_dim = 128, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 13, init = 'uniform', activation = 'sigmoid'))


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

#handling test data

dataset2 = pd.read_csv('test_data.csv')
corpus_test = []

for i in range(0, 3000):
    review = re.sub('[^a-zA-Z]', ' ', dataset2['content'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus_test.append(review)



X_test = cv.fit_transform(corpus_test).toarray()

y_pred = classifier.predict(X_test)

dataset3 = pd.read_csv('sample_submission.csv')
#y_test = dataset3.iloc[:,1].values
y_test = cv.fit_transform(dataset3['sentiment']).toarray()

# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
