from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from pickle import load
import numpy as np
import random

def is_spam(status):
    spam = 1 if status == ',spam\n' else 0
    return spam

#messages = load_structure('messages.pkl')
messages = []
y = []
data = open('SMS_Spam_Corpus_big.txt', encoding = 'latin1')
lines = data.readlines()
for line in lines:
    comma = line.rfind(',')
    messages.append(line[:comma])
    y.append(is_spam(line[comma:]))
y = np.array(y)
vectorizer = CountVectorizer()
vectors = vectorizer.fit_transform(messages)
transformer = TfidfTransformer()
vectors = transformer.fit_transform(vectors)
#print(vectors.todense())
# Split in training and testing data
X,X_test,y,y_test = train_test_split(vectors.todense(),y, test_size = 0.7, random_state = 42)
lr = LogisticRegression()
lr.fit(X,y)
train_score = lr.score(X,y)
predictions = lr.predict(X_test)
test_score = lr.score(X_test,y_test)
print('Training score: ', train_score)
print('Testing score: ', test_score)
print('Coefficients:', lr.coef_)

i = 0
for p in predictions:
    print('Expected: ',y_test[i], ' Predicted: ', p)
    i += 1
