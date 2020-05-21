# -*- coding: utf-8 -*-
import os
import numpy as np



def load_glove(filename,tokenizer_word_index,EMBEDDING_DIM):
    
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


