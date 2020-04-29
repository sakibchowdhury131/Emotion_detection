# -*- coding: utf-8 -*-

import string
import re
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

for_embedding = list()

for line in dfNew['phrase']:
    ls = word_tokenize(line)
    for_embedding.append(ls)
    


# train word2vec model
EMBEDDING_DIM = 100
model = gensim.models.Word2Vec(sentences = for_embedding, size = EMBEDDING_DIM, window = 5, workers = 4, min_count = 4,sg=1)
# vec_king = model.wv['the']


# for i, word in enumerate(model.wv.vocab):
#     if i == 10:
#         break
#     print(word)
    
# model.most_similar ('dad')


# #some similarity fun
# print(model.wv.similarity('woman', 'man'), model.wv.similarity('man', 'elephant'))

