import pandas as pd
import re
from nltk.tokenize import word_tokenize
import gensim
import os
import numpy as np
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,GRU,Flatten
from keras.initializers import Constant
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from keras.models import model_from_json
from keras.layers import Conv1D, MaxPooling1D

def filter_texts (dataset): # filters texts           
    for i in range(0, dataset.shape[0]):
        comment = re.sub('[^a-zA-Z0-9\s@]', ' ', dataset['text'][i])
        comment = comment.lower()
        comment = comment.split()
        comment = [j for j in comment if len(j) > 1]
        comment = ' '.join(word for word in comment if not word.startswith('@'))
        dataset['text'][i] = comment
    return dataset

dataset = pd.read_csv('Tweets.csv')
dataset = dataset.iloc[:,[1,10]]

dataset = filter_texts(dataset)

texts = dataset['text'].values.tolist()
sentiment = dataset['airline_sentiment'].values.tolist()


keywords = []

for i in range (0,14640):
    if not(sentiment[i] in keywords):
        keywords.append(sentiment[i])


review_lines = list()

for line in texts:
    words = word_tokenize(line)
    review_lines.append(words)


# train word2vec model
EMBEDDING_DIM = 100
model = gensim.models.Word2Vec(sentences = review_lines, size = EMBEDDING_DIM, window = 5, workers = 4, min_count = 4,sg=1)



filename = 'pretrained_word2vec_sg_twitter.txt'
model.wv.save_word2vec_format(filename, binary = False)


embedding_index = {}


f = open(os.path.join('','pretrained_word2vec_sg_twitter.txt'), encoding = "utf-8")


for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:])
    embedding_index[word] = coefs


f.close()



tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(review_lines)
sequences = tokenizer_obj.texts_to_sequences(review_lines)



#pad sequence
word_index = tokenizer_obj.word_index


max_length = 300
review_pad = pad_sequences(sequences, maxlen = max_length)



num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words,EMBEDDING_DIM))


for word,i in word_index.items():
    if i> num_words:
        continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

kernel_size = 5
filters = 40
pool_size = 4

#define model
model_nn = Sequential()
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer = Constant(embedding_matrix),
                            input_length = max_length,
                            trainable = False)


model_nn.add (embedding_layer)


model_nn.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model_nn.add(MaxPooling1D(pool_size=pool_size))
model_nn.add(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))
model_nn.add(Dense(128, activation = 'relu'))
model_nn.add(GRU(units = 48, dropout = 0.2, recurrent_dropout = 0.2))
model_nn.add(Dense(3, activation = 'sigmoid'))
model_nn.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model_nn.summary()

y = pd.get_dummies(dataset['airline_sentiment']).values
[print(dataset['airline_sentiment'][i], y[i]) for i in range(0,15)]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(review_pad, y, test_size = 0.20, random_state = 0)


history = model_nn.fit(X_train, y_train, batch_size = 32, epochs = 10 , validation_data = (X_test,y_test), verbose = 1)


y_pred = model_nn.predict(X_test)
y_pred = y_pred>0.5


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


print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('F-measure: %.3f' % f1_score(y_test, y_pred))


# serialize model to JSON
model_json = model_nn.to_json()
with open("twitter_lstm_cbow.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model_nn.save_weights("twitter_lstm_cbow.h5")
print("Saved model to disk")
 
# later...
 
# load json and create model
json_file = open('twitter_lstm_cbow.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("twitter_lstm_cbow.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))# -*- coding: utf-8 -*-

