# -*- coding: utf-8 -*-

from gensim import utils
from read_stanford_sentiment_treebank import read_data
import numpy as np
import gensim.models



#read the dataset
dataset = read_data ('/media/sakib/alpha/work/EmotionDetectionDir/word2vec_embedding/stanfordSentimentTreebank') 

#create sentences
class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""

    def __iter__(self):
        for line in dataset['Phrase']:
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)
            


sentences = MyCorpus()


#train model
w2v = gensim.models.Word2Vec(sentences=sentences, min_count = 1 , size = 100,iter = 10, workers = 4)

# vec_king = w2v.wv['brilliant']

# for i, word in enumerate(w2v.wv.vocab):
#     if i == 10:
#         break
#     print(word)
    
# w2v.most_similar ('that')


# print(w2v.wv.index2word[0], w2v.wv.index2word[1], w2v.wv.index2word[2])
# vocab_size = len(w2v.wv.vocab)
# print(w2v.wv.index2word[vocab_size - 1], w2v.wv.index2word[vocab_size - 2], w2v.wv.index2word[vocab_size - 3])

# find the index of the 2nd most common word ("of")
# print('Index of "of" is: {}'.format(w2v.wv.vocab['of'].index))



# some similarity fun
# print(w2v.wv.similarity('woman', 'man'), w2v.wv.similarity('man', 'elephant'))

# what doesn't fit?
# print(w2v.wv.doesnt_match("you me elephant i they ".split()))


#saving the model
w2v.save("w2v_model")

#loading the saved model 

# model = gensim.models.Word2Vec.load("w2v_model")


# convert the wv word vectors into a numpy matrix that is suitable for insertion

# into our TensorFlow and Keras models

# vector_dim = 100
# embedding_matrix = np.zeros((len(model.wv.vocab),vector_dim ))
# for i in range(len(model.wv.vocab)):
#     embedding_vector = model.wv[model.wv.index2word[i]]
#     if embedding_vector is not None:
#         embedding_matrix[i] = embedding_vector