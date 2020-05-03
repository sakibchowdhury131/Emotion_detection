import pandas as pd
import re
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def filter_texts (dataset): # filters texts           
    for i in range(0, dataset.shape[0]):
        comment = re.sub('[^a-zA-Z0-9\s@]', ' ', dataset['text'][i])
        comment = comment.lower()
        comment = comment.split()
        comment = [j for j in comment if len(j) > 1]
        comment = ' '.join(word for word in comment if not word.startswith('@'))
        dataset['text'][i] = comment
    return dataset


dataset = pd.read_csv('Tweets.csv')
dataset = dataset.iloc[:,[1,10]]


dataset = filter_texts(dataset)


lines = dataset['text'].values.tolist()
sentiment = dataset['airline_sentiment']


from sklearn.feature_extraction.text import TfidfVectorizer
# Create feature vectors
vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)
X = vectorizer.fit_transform(dataset['text'])
y = dataset['airline_sentiment']



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('F-measure: %.3f' % f1_score(y_test, y_pred))


