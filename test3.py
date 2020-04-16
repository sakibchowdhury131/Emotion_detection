# importing libraries
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences



def filter_texts (dataset): # filters texts
            
    for i in range(0, dataset.shape[0]):
        comment = re.sub('[^a-zA-Z0-9\s@]', ' ', dataset['content'][i])
        
        comment = comment.lower()
        comment = comment.split()
        
        comment = [j for j in comment if len(j) > 1]
    
        comment = ' '.join(word for word in comment if not word.startswith('@'))
        dataset['content'][i] = comment
    return dataset
        
    


dataset = pd.read_csv('train_data.csv')


# analyzing the keywords

y = dataset.iloc[:, 0].values

keywords = []

for i in range (0,30000):
    if not(y[i] in keywords):
        keywords.append(y[i])
        

for i in range(0,30000):
    if (y[i]=='empty' or y[i]=='sadness' or y[i]=='worry' or y[i]=='hate' or y[i]=='anger'):
        y[i] = 'negative'
        
    elif y[i] =='neutral' :
        y[i] = 'neutral'
        
    else:
        y[i] = 'positive'
        
        
#filtering the texts
        
dataset = filter_texts(dataset)


# visualizing final dataset
    
dataset['sentiment'].value_counts().sort_index().plot.bar()


dataset['content'].str.len().plot.hist()

#tokenizing
    

tokenizer = Tokenizer(num_words=5000, split=" ")
tokenizer.fit_on_texts(dataset['content'].values)

X = tokenizer.texts_to_sequences(dataset['content'].values)
X = pad_sequences(X) # padding our text vector so they all have the same length


# Model Designing
model = Sequential()
model.add(Embedding(5000, 256, input_length=X.shape[1]))
model.add(Dropout(0.3))
model.add(LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))
model.add(LSTM(256, dropout=0.3, recurrent_dropout=0.2))
model.add(Dense(3, activation='sigmoid'))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# encoding the sentiments


y = pd.get_dummies(dataset['sentiment']).values
[print(dataset['sentiment'][i], y[i]) for i in range(0,15)]

# test train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



# fitting the model
batch_size = 25
epochs = 23

model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

model.save('sentiment_analysis.h5')


#prediction

predictions = model.predict(X_test)
[print(dataset['content'][i], predictions[i], y_test[i]) for i in range(0, 5)]


#visualizing results

pos_count, neu_count, neg_count = 0, 0, 0
real_pos, real_neu, real_neg = 0, 0, 0
for i, prediction in enumerate(predictions):
    if np.argmax(prediction)==2:
        pos_count += 1
    elif np.argmax(prediction)==1:
        neu_count += 1
    else:
        neg_count += 1
    
    if np.argmax(y_test[i])==2:
        real_pos += 1
    elif np.argmax(y_test[i])==1:    
        real_neu += 1
    else:
        real_neg +=1

print('Positive predictions:', pos_count)
print('Neutral predictions:', neu_count)
print('Negative predictions:', neg_count)
print('Real positive:', real_pos)
print('Real neutral:', real_neu)
print('Real negative:', real_neg)
