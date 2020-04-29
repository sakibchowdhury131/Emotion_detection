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
from keras.layers import Dense,Embedding,LSTM,GRU
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
import matplotlib.pyplot as plt
dataset = read_data ('/media/sakib/alpha/work/EmotionDetectionDir/pretrained embedding/word2vec_embedding/stanfordSentimentTreebank') 
# binarizing sentiments
dataset['sentiment_values'] = pd.to_numeric(dataset['sentiment_values'], downcast = 'float')
dataset['sentiment_values'] = (dataset['sentiment_values'] >= 0.4).astype(float)
review_lines = list()
lines = dataset['Phrase'].values.tolist()


for line in lines:
    tokens = word_tokenize(line)
    token = [w.lower() for w in tokens]
    table = str.maketrans('','',string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    review_lines.append(words)

    

# train word2vec model
EMBEDDING_DIM = 100
model = gensim.models.Word2Vec(sentences = review_lines, size = EMBEDDING_DIM, window = 5, workers = 4, min_count = 4,sg=1)

#vocab size
words = list(model.wv.vocab)

# model.wv.most_similar('father')

filename = 'pretrained_word2vec_sg_stanford.txt'
model.wv.save_word2vec_format(filename, binary = False)


embedding_index = {}

f = open(os.path.join('','pretrained_word2vec_sg_stanford.txt'), encoding = "utf-8")

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:])
    embedding_index[word] = coefs

f.close()

# vectorize the text samples into a 2D integer tensor

tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(review_lines)
sequences = tokenizer_obj.texts_to_sequences(review_lines)

#pad sequence
word_index = tokenizer_obj.word_index

max_length = 300
review_pad = pad_sequences(sequences, maxlen = max_length)
sentiment = dataset ['sentiment_values'].values

num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words,EMBEDDING_DIM))

for word,i in word_index.items():
    if i> num_words:
        continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        
print(num_words)


#define model
model_nn = Sequential()
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer = Constant(embedding_matrix),
                            input_length = max_length,
                            trainable = False)

model_nn.add (embedding_layer)
model_nn.add(GRU(units = 32, dropout = 0.2, recurrent_dropout = 0.2))
model_nn.add(Dense(1, activation = 'sigmoid'))

model_nn.compile(loss= 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model_nn.summary()

VALIDATION_SPLIT = 0.2

indices = np.arange(review_pad.shape[0])
np.random.shuffle(indices)
review_pad = review_pad[indices]
sentiment = sentiment[indices]
num_validation_samples = int(VALIDATION_SPLIT * review_pad.shape[0])

X_train = review_pad[: -num_validation_samples]
y_train = sentiment[:-num_validation_samples]
X_test = review_pad[-num_validation_samples:]
y_test = sentiment[-num_validation_samples:]


history = model_nn.fit(X_train,y_train,batch_size = 128, epochs = 25 , validation_data = (X_test,y_test),verbose = 1)

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


model.save('GRU+SG.h5')