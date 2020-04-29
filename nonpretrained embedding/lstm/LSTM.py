# -*- coding: utf-8 -*-

import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from read_stanford_sentiment_treebank import read_data
import gensim
import os
import numpy as np
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,GRU,Flatten
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
import matplotlib.pyplot as plt
from keras import optimizers
from keras import regularizers
from sklearn.model_selection import train_test_split

dataset = read_data ('/media/sakib/alpha/work/EmotionDetectionDir/nonpretrained embedding/lstm/stanfordSentimentTreebank') 
dataset['sentiment_values'] = pd.to_numeric(dataset['sentiment_values'], downcast = 'float')
dataset['sentiment_values'] = (dataset['sentiment_values'] >= 0.4).astype(float)
y = dataset['sentiment_values']



tokenizer_obj = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True)
tokenizer_obj.fit_on_texts(dataset['Phrase'])

sequences = tokenizer_obj.texts_to_sequences(dataset['Phrase'])
max_length = 300
X = pad_sequences(sequences, maxlen = max_length)


model = Sequential()
model.add(Embedding(2000, 128, input_length=X.shape[1]))
model.add(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))
model.add(Flatten())
model.add(Dense(1, activation='softmax',kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)))
adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# fitting the model
batch_size = 128
epochs = 25

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle = True, validation_data = (X_test,y_test))



# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


model.save('LSTM.h5')
