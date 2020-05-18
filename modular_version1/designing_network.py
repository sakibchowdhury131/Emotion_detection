# -*- coding: utf-8 -*-
"""
Created on Sat May  9 22:34:44 2020

@author: sakib

"""


from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM
from keras.initializers import Constant
from keras.models import model_from_json
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import csv


def model_architecture_word2vec(embedding_matrix,num_words,EMBEDDING_DIM = 100, max_length = 150):
    model_nn = Sequential()
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                embeddings_initializer = Constant(embedding_matrix),
                                input_length = max_length,
                                trainable = False)
    
    
    model_nn.add (embedding_layer)
    model_nn.add(LSTM(100, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))
    model_nn.add(LSTM(100, dropout=0.3, recurrent_dropout=0.2))
    model_nn.add(Dense(3, activation = 'sigmoid'))
    
    model_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_nn.summary()
    return model_nn




def model_architecture_glove(embedding_matrix,num_words,EMBEDDING_DIM = 100,max_length = 150):
    model_nn = Sequential()
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                embeddings_initializer = Constant(embedding_matrix),
                                input_length = max_length,
                                trainable = False)
    
    
    model_nn.add (embedding_layer)
    model_nn.add(LSTM(100, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))
    model_nn.add(LSTM(100, dropout=0.3, recurrent_dropout=0.2))
    model_nn.add(Dense(3, activation = 'sigmoid'))
    
    model_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_nn.summary()
    return model_nn



def model_architecture_sswe(embedding_matrix,num_words,EMBEDDING_DIM = 50, max_length = 150):
    model_nn = Sequential()
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                embeddings_initializer = Constant(embedding_matrix),
                                input_length = max_length,
                                trainable = False)

    model_nn.add (embedding_layer)
    model_nn.add(LSTM(100, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))
    model_nn.add(LSTM(100, dropout=0.3, recurrent_dropout=0.2))
    model_nn.add(Dense(3, activation = 'sigmoid'))
    
    model_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_nn.summary()
    return model_nn
 

def fit_network(model, X_train, X_test, y_train, y_test,batch_size = 32, epochs = 10 ,verbose = 1):
    history = model.fit(X_train,y_train,batch_size = batch_size, epochs = epochs , validation_data = (X_test,y_test),verbose = verbose)
    return model, history


def save_network_model(model_nn,modelname,directory):
    # serialize model to JSON
    model_json = model_nn.to_json()
    with open(directory+'/'+ modelname+'.json', "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model_nn.save_weights(directory+'/'+ modelname+'.h5')
    print("Saved model to disk")
    
    
    
def load_network_model(directory,jsonfile, h5file):
    json_file = open(directory+'/'+jsonfile, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(directory+'/'+h5file)
    print("Loaded model from disk")
    return loaded_model

def analyze_performance(model_name,loaded_model,X_test, y_test):
    
    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    score = loaded_model.evaluate(X_test, y_test, verbose=1)
    print ('accuracy: ', score[1]*100)
    y_pred = loaded_model.predict(X_test)
    y_pred= y_pred>0.5
    
    neg_pre = precision_score(y_test[:,0], y_pred[:,0])
    neu_pre = precision_score(y_test[:,1], y_pred[:,1])
    pos_pre = precision_score(y_test[:,2], y_pred[:,2])
    
    neg_rec = recall_score(y_test[:,0], y_pred[:,0])
    neu_rec = recall_score(y_test[:,1], y_pred[:,1]) 
    pos_rec = recall_score(y_test[:,2], y_pred[:,2])
    
    neg_f1 = f1_score(y_test[:,0], y_pred[:,0])
    neu_f1 = f1_score(y_test[:,1], y_pred[:,1])
    pos_f1 = f1_score(y_test[:,2], y_pred[:,2])
    
    with open('report1.csv', 'a+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([model_name, score[1]*100, neg_f1, neg_pre, neg_rec, neu_f1, neu_pre,neu_rec,pos_f1,pos_pre,pos_rec])

        
    print('negative precision: ', neg_pre )
    print('neutral precision: ',  neu_pre)
    print('positive precision: ', pos_pre)
    
    print('negative recall: ', neg_rec)
    print('neutral recall: ', neu_rec)
    print('positive recall: ', pos_rec)
    
    print('negative f1_score: ', neg_f1 )
    print('neutral f1_score: ',  neu_f1)
    print('positive f1_score: ', pos_f1)
    
    
    


