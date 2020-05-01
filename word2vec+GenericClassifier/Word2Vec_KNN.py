from read_stanford_sentiment_treebank import read_data
import pandas as pd
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
import gensim
import numpy as np
import re
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

dataset = read_data ('/media/sakib/alpha/work/EmotionDetectionDir/stanfordSentimentTreebank') 
# binarizing sentiments
dataset['sentiment_values'] = pd.to_numeric(dataset['sentiment_values'], downcast = 'float')
dataset['sentiment_values'] = (dataset['sentiment_values'] >= 0.4).astype(float)
review_lines = list()
lines = dataset['Phrase'].values.tolist()

sentiment = dataset['sentiment_values'].values.tolist()


for line in lines:
    review = re.sub('[^a-zA-Z]', ' ', line)
    review = review.lower()
    review_lines.append(review)


df = pd.DataFrame(
    {'phrase': review_lines,
     'sentiment': sentiment
     })

df['phrase'] = df['phrase'].str.lstrip()


filters = df['phrase'] != ""
dfNew = df[filters]
lines = dfNew['phrase'].values.tolist()


review_lines = list()
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
model = gensim.models.Word2Vec(sentences = review_lines, size = EMBEDDING_DIM, window = 5, workers = 4, min_count = 1,sg=1)





#vocab size
words = list(model.wv.vocab)
word_vectors = model.wv
# building the feature matrix
feature = [[0]*2*EMBEDDING_DIM]*len(review_lines)
for i in range (len(review_lines)):
    if not len(review_lines[i])==0:
        vector = [0]*len(review_lines[i])
        for j in range (0,len(review_lines[i])):
            if review_lines[i][j] in word_vectors:
                vector[j] = model.wv[review_lines[i][j]] 
            else :
                vector[j] = [0]*100
            
            
            # Get the minimum values of each column i.e. along axis 0
        minInColumns = np.amin(vector, axis=0)
        maxInColumns = np.amax(vector, axis=0)
        
        
        if not (minInColumns.shape == () or maxInColumns.shape == ()):
            feature[i] = np.concatenate((minInColumns,maxInColumns),axis = 0)
            
sentiment = dfNew['sentiment'].tolist()
i = 0
while i!=len(feature):
    
    flag = 0
    for j in range (len(feature[i])):
        if feature[i][j]!=0:
            flag = 1
            break
    
    if flag == 0:
        del feature[i]
        del sentiment[i]
        
        
    i = i+1
    
    
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(feature, sentiment, test_size = 0.20, random_state = 0)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train.toarray(), y_train)

y_pred = classifier.predict(X_test)


print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('F-measure: %.3f' % f1_score(y_test, y_pred))