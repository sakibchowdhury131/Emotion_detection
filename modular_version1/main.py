#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 22:34:44 2020

@author: sakib

"""


#change base directory
import os

base_path=r'/media/sakib/alpha/work/EmotionDetectionDir/git'
working_directory = '/media/sakib/alpha/work/EmotionDetectionDir/git'
data_folder = r'/media/sakib/alpha/work/EmotionDetectionDir/git/data'
embedding_folder = r'/media/sakib/alpha/work/EmotionDetectionDir/git/embeddings'
models_folder = r'/media/sakib/alpha/work/EmotionDetectionDir/git/models'
model_directory = working_directory+'/'+'models'


def change_base_dir(base_dir_path):
    """ Change the working directopry of the code"""
    
    if not os.path.exists(base_dir_path):
        print ('creating directory', base_dir_path)
        os.makedirs(base_dir_path)
    print ('Changing base directory to ', base_dir_path)
    os.chdir(base_dir_path)
 
      

# base_folder='data'
# base_dir_path=base_path+'/'+base_folder
# change_base_dir(base_dir_path)
# download_data.extract_file('/media/sakib/alpha/work/EmotionDetectionDir/git/data/glove.6B.zip')



change_base_dir(base_path)
if not os.path.exists(data_folder):
        os.makedirs(data_folder)
if not os.path.exists(embedding_folder):
        os.makedirs(embedding_folder)
if not os.path.exists(models_folder):
        os.makedirs(models_folder)
  
#loading necessary files
from load_preprocess import load_data,load_data_embedding
from load_preprocess import preprocess_data
from load_preprocess import read_labels
import word2vec
import sswe
import glove_file
import designing_network
import download_data


#importing libraries
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
import pandas as pd



# Download data

print('Checking requirements:')
download_data.download_data(base_path = base_path)

#loading data
print('Step1: Loading Embedding training Dataset...')


dataset_embedding = load_data_embedding(working_directory+'/data/'+'training.1600000.processed.noemoticon.csv')
print('Step2: Shuffling data...')
dataset_embedding = dataset_embedding.sample(frac=1) # reshuffling the data
print('Step3: Preprocessing the texts...')
texts = preprocess_data(dataset_embedding)
labels = read_labels(dataset_embedding)


# building word2vec model
print('Step4: Building word2vec model...')
EMBEDDING_DIM = 100
w2v = word2vec.create_word2vec(directory = working_directory+'/'+'embeddings',texts = texts ,min_count = 1,EMBEDDING_DIM = EMBEDDING_DIM)



#building sswe model
print('Step5: Building sswe model...')
sswe_model,training_word_index = sswe.sswe_model(texts, labels)
embedding_weights, word_indices_df, merged = sswe.save_sswe(sswe_model,training_word_index,directory = working_directory+'/'+'embeddings')
print('Embedding Layers are trained.')



# loading twitter_dataset_small
print('Step6: Loading Twitter Dataset')
dataset = load_data(working_directory+'/'+'data'+'/'+'Tweets.csv')
texts = preprocess_data(dataset)
y = pd.get_dummies(dataset['Label']).values



#tokenizing
print ('Step7: Tokenizing...')

tokens = []
for line in texts:
    words = word_tokenize(line)
    tokens.append(words)



tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(tokens)
sequences = tokenizer_obj.texts_to_sequences(tokens)


#padding
print ('Step8: Padding...')
tokenizer_word_index = tokenizer_obj.word_index
max_length = 150
review_pad = pad_sequences(sequences, maxlen = max_length)



# train test split
print('Step9: train and test set generation...')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(review_pad, y, test_size = 0.20, random_state = 0)


# word2vec Embedding Matrix
print('Step10: Generating word2vec embedding matrix...')
num_words = len(tokenizer_word_index) + 1
embedding_matrix_w2v = word2vec.load_word2vec(working_directory+'/'+'embeddings'+'/'+'embeddings_w2v.txt', tokenizer_word_index=tokenizer_word_index, EMBEDDING_DIM=EMBEDDING_DIM)


# training the word2vec model with lstm
print('Step11: designing lstm+w2v model...')

w2v_lstm = designing_network.model_architecture_word2vec(embedding_matrix_w2v, num_words,EMBEDDING_DIM = EMBEDDING_DIM , max_length = max_length)
w2v_lstm, history = designing_network.fit_network(w2v_lstm, X_train, X_test, y_train, y_test)
designing_network.save_network_model(w2v_lstm, modelname = 'w2v_lstm',directory = model_directory)
# loaded_model = designing_network.load_network_model( directory = working_directory+'/'+'models', jsonfile = 'w2v_lstm.json', h5file = 'w2v_lstm.h5')
 

# sswe embedding matrix
print('Step12: Generating sswe embedding matrix...')
sswe_embedding_filename = working_directory+'/'+'embeddings'+'/'+'embeddings_sswe.tsv'
embedding_matrix_sswe = sswe.load_sswe(filename = sswe_embedding_filename, tokenizer_word_index = tokenizer_word_index, EMBEDDING_DIM = 50)


# training the sswe model with lstm
print('Step13: designing lstm+sswe model...')
sswe_lstm = designing_network.model_architecture_sswe(embedding_matrix_sswe, num_words,EMBEDDING_DIM = 50 , max_length = max_length)
sswe_lstm, history = designing_network.fit_network(sswe_lstm, X_train, X_test, y_train, y_test)
designing_network.save_network_model(sswe_lstm, modelname = 'sswe_lstm',directory = model_directory)
# loaded_model = designing_network.load_network_model( directory = working_directory+'/'+'models', jsonfile = 'w2v_lstm.json', h5file = 'w2v_lstm.h5')


# glove embedding matrix
print ('Step 14: Generating glove embedding matrix...')
embedding_matrix_glove = glove_file.load_glove(working_directory+'/'+'data'+'/'+'glove.6B.100d.txt', tokenizer_word_index=tokenizer_word_index, EMBEDDING_DIM=EMBEDDING_DIM)



# training the glove model with lstm
print('Step15: designing lstm+w2v model...')
glove_lstm = designing_network.model_architecture_glove(embedding_matrix_glove, num_words,EMBEDDING_DIM = EMBEDDING_DIM , max_length = max_length)
glove_lstm, history = designing_network.fit_network(glove_lstm, X_train, X_test, y_train, y_test)
designing_network.save_network_model(glove_lstm, modelname = 'glove_lstm',directory = model_directory)
# loaded_model = designing_network.load_network_model( directory = working_directory+'/'+'models', jsonfile = 'glove_lstm.json', h5file = 'glove_lstm.h5')

