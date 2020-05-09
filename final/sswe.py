# -*- coding: utf-8 -*-
"""
Created on Sat May  9 22:34:44 2020

@author: sakib

"""


import pandas as pd
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding
from keras.layers.pooling import MaxPooling1D
from keras.layers.convolutional import Convolution1D
from keras import optimizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import num2words
import os 


def sswe_model(train_texts,train_labels):
    random_seed=1
    np.random.seed(random_seed)

    
    max_sequence_length = 15 # each sentence of the input should be padded to have at least this many tokens
    embedding_dim 		= 50 # Embedding layer size
    no_filters			= 15 # No of filters for the convolution layer
    filter_size			= 5  # Filter size for the convolution layer
    trainable 			= True # flag specifying whether the embedding layer weights should be changed during the training or not
    batch_size 			= 1024*6 # batch size can be increased to have better gpu utilization
    #batch_size 			= 64 # batch size can be increased to have better gpu utilization
    no_epochs 			= 5 # No of training epochs
    
    
    
    pos_emoticons=["(^.^)","(^-^)","(^_^)","(^_~)","(^3^)","(^o^)","(~_^)","*)",":)",":*",":-*",":]",":^)",":}",
                   ":>",":3",":b",":-b",":c)",":D",":-D",":O",":-O",":o)",":p",":-p",":P",":-P",":Þ",":-Þ",":X",
                   ":-X",";)",";-)",";]",";D","^)","^.~","_)m"," ~.^","<=8","<3","<333","=)","=///=","=]","=^_^=",
                   "=<_<=","=>.<="," =>.>="," =3","=D","=p","0-0","0w0","8D","8O","B)","C:","d'-'","d(>w<)b",":-)",
                   "d^_^b","qB-)","X3","xD","XD","XP","ʘ‿ʘ","❤","💜","💚","💕","💙","💛","💓","💝","💖","💞",
                   "💘","💗","😗","😘","😙","😚","😻","😀","😁","😃","☺","😄","😆","😇","😉","😊","😋","😍",
                   "😎","😏","😛","😜","😝","😮","😸","😹","😺","😻","😼","👍"]
    
    neg_emoticons=["--!--","(,_,)","(-.-)","(._.)","(;.;)9","(>.<)","(>_<)","(>_>)","(¬_¬)","(X_X)",":&",":(",":'(",
                   ":-(",":-/",":-@[1]",":[",":\\",":{",":<",":-9",":c",":S",";(",";*(",";_;","^>_>^","^o)","_|_",
                   "`_´","</3","<=3","=/","=\\",">:(",">:-(","💔","☹️","😌","😒","😓","😔","😕","😖","😞","😟",
                   "😠","😡","😢","😣","😤","😥","😦","😧","😨","😩","😪","😫","😬","😭","😯","😰","😱","😲",
                   "😳","😴","😷","😾","😿","🙀","💀","👎"]
    

    
    emoticonsDict = {}
    for i,each in enumerate(pos_emoticons):
        emoticonsDict[each]=' POS_EMOTICON_'+num2words.num2words(i).upper()+' '
        
    for i,each in enumerate(neg_emoticons):
        emoticonsDict[each]=' NEG_EMOTICON_'+num2words.num2words(i).upper()+' '
        
    # use these three lines to do the replacement
  

    # Loading Training and Validation Data
    labels = []

    
    print (len(train_labels), len(train_texts))
    print ("Using Keras tokenizer to tokenize and build word index")
    tokenizer = Tokenizer(lower=True, filters='\n\t?"!') 
    train_texts=[each for each in train_texts]
    tokenizer.fit_on_texts(train_texts)
    sorted_voc = [wc[0] for wc in sorted(tokenizer.word_counts.items(),reverse=True, key= lambda x:x[1]) ]
    tokenizer.word_index = dict(list(zip(sorted_voc, list(range(2, len(sorted_voc) + 2)))))
    tokenizer.word_index['<PAD>']=0
    tokenizer.word_index['<UNK>']=1
    word_index = tokenizer.word_index
    vocab_size=len(tokenizer.word_index.keys())
    
    print ('Size of the vocab is', vocab_size)
    
    
    
    print ('Padding sentences and shuffling the sswe train data')
    sequences = tokenizer.texts_to_sequences(train_texts)
    
    #Pad the sentences to have consistent length
    data = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
    labels = to_categorical(np.asarray(train_labels))
    indices = np.arange(len(labels))
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    
    train_x, valid_x, train_y, valid_y=train_test_split(data, labels, test_size=0.2, random_state=random_seed)
    train_x=np.array(train_x).astype('float32')
    valid_x=np.array(valid_x).astype('float32')
    train_y=np.array(train_y)
    valid_y=np.array(valid_y)
    training_word_index=tokenizer.word_index.copy()
    
    
    print ('Initializing the model')
    mcp = ModelCheckpoint('./model_chkpoint', monitor="val_acc", save_best_only=True, save_weights_only=False)
    
    #Creating network
    model = Sequential()
    model.add(Embedding(len(word_index)+2,
                                embedding_dim,
                                input_length=max_sequence_length,
                                trainable=trainable, name='embedding'))
    model.add(Convolution1D(no_filters, filter_size, activation='relu'))
    model.add(MaxPooling1D(max_sequence_length - filter_size))
    model.add(Flatten())
    model.add(Dense(no_filters, activation='tanh'))
    model.add(Dense(len(labels[0]), activation='softmax'))
    
    optim=optimizers.Adam(lr=0.1, )
    model.compile(loss='categorical_crossentropy',
                  optimizer=optim,
                  metrics=['acc'])
    model.summary()
    

    model.fit(train_x, train_y,nb_epoch=no_epochs, batch_size=batch_size,validation_data=(valid_x, valid_y),callbacks=[mcp])
    return model,training_word_index



# Exporting the Embedding Matrix and Vocabulary
    
def save_sswe(model,training_word_index):
    """ export embeddings to file"""
    model_identifier = 'sswe'
    directory = '/media/sakib/alpha/work/EmotionDetectionDir/Final_codes/embeddings'
    embedding_weights=pd.DataFrame(model.layers[0].get_weights()[0]).reset_index()
    word_indices_df=pd.DataFrame.from_dict(training_word_index,orient='index').reset_index()
    word_indices_df.columns=['word','index']
    print (word_indices_df.shape,embedding_weights.shape)
    merged=pd.merge(word_indices_df,embedding_weights)
    print (merged.shape)
    merged=merged[[each for each in merged.columns if each!='index']]    
    merged.to_csv(directory+'/'+'embeddings_{}.tsv'.format(model_identifier), sep='\t', 
              index=False, header=False,float_format='%.6f',encoding='utf-8')
    return embedding_weights, word_indices_df, merged


def load_sswe(filename,tokenizer_word_index,EMBEDDING_DIM):
    
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