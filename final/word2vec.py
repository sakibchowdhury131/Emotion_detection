#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 22:34:44 2020

@author: sakib
"""
from nltk.tokenize import word_tokenize
import gensim
import os
import numpy as np

def create_word2vec(texts,EMBEDDING_DIM = 100,window = 5,workers = 4, min_count = 4,sg=1):
    tokens = []
    for line in texts:
        words = word_tokenize(line)
        tokens.append(words)
    
    model = gensim.models.Word2Vec(sentences = tokens, size = EMBEDDING_DIM, window = 5, workers = 4, min_count = 4,sg=1)
    print('Saving word2vec model in the disk')
    directory = '/media/sakib/alpha/work/EmotionDetectionDir/Final_codes/embeddings'
    filename = 'embeddings_w2v.txt'
    model.wv.save_word2vec_format(directory+'/'+filename, binary = False)

    return model

def load_word2vec(filename,tokenizer_word_index,EMBEDDING_DIM):
    
    embedding_index = {}
    
    
    f = open(os.path.join('',filename), encoding = "utf-8")
    
    
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:])
        embedding_index[word] = coefs
    
    
    f.close()
    
    
    num_words = len(tokenizer_word_index) + 1
    embedding_matrix = np.zeros((num_words,EMBEDDING_DIM))
    for word,i in tokenizer_word_index.items():
        if i> num_words:
            continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            
    return embedding_matrix

