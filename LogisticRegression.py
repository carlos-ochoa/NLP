import numpy as np
import matplotlib.pyplot as plt
from pickle import load, dump
import nltk
from nltk.corpus import nps_chat
from nltk.stem import WordNetLemmatizer

# Functions to save files
def save_structure(structure,filename):
    file = open(filename,'wb')
    dump(structure,file)
    file.close()

def load_structure(filename):
    file = open(filename,'rb')
    structure = load(file)
    file.close()
    return structure

class LogisticRegression():

    def __init__(self,params,X,y):
        self.X, self.Y = X,y.T
        self.l = 0
        self.total_loss = []
        self.w = np.array([np.zeros(params)])

    def hypothesis(self):
        #print(self.w)
        #print(self.X[0])
        return np.dot(self.X, self.w.T)

    def sigmoid(self,z):
        #print(z[0])
        #print(z.shape)
        return 1 / (1 + np.exp(-z))

    def predict(self,z):
        p = []
        pre = self.sigmoid(z)
        for v in pre:
            p.append(1 if v > 0.5 else 0)
        p = np.array([np.array(p)])
        return p.T

    def loss(self,h):
        epsilon = 1e-5
        return -1 * np.mean(self.Y * np.log(h + epsilon) + (1 - self.Y) * np.log(1 - h + epsilon))

    def gradient_descent(self, alpha = 0.00001):
        error = self.sigmoid(self.hypothesis()) - self.Y
        m = self.Y.size
        dw = []
        i = 0
        #dw = np.dot(error.T,self.X) * (1/m)
        for x in self.X:
            dw.append(error[i] * x)
            i += 1
        dw = np.array(dw)
        dw = np.sum(dw, axis = 0) / len(self.X)
        dw = np.array([dw])
        self.w -= alpha * dw

    def train(self, iter):
        for i in range(iter):
            predictions = self.sigmoid(self.hypothesis())
            #print(predictions[0])
            self.l = self.loss(predictions)
        #    print(self.l)
            self.total_loss.append(self.l)
            self.gradient_descent()
            print('Current loss ', self.l)
        save_structure(self.w, 'w.pkl')
        plt.plot(range(iter), self.total_loss)
        plt.show()

    def test(self,X,messages):
        i = 0
        mess, is_spam = [], []
        predictions = self.predict(self.hypothesis())
        for x in X:
            print(x)
            mess.append(messages[i][0])
            is_spam.append(predictions[i][0])
            print(mess[i], ' Predicted : ', predictions[i][0])
            i += 1
        #output.to_csv('output.csv', index = False)

def create_tagger():
    chat_tags = nps_chat.tagged_posts()
    t0 = nltk.DefaultTagger('NN')
    t1 = nltk.UnigramTagger(chat_tags, backoff = t0)
    t2 = nltk.BigramTagger(chat_tags, backoff = t1)
    return t2

def tagger(tokens,t2):
    tags = t2.tag(tokens)
    print('tag')
    return tags

def lemmatizer(tokens,lem):
    lem_tokens = []
    for (word,pos_t) in tokens:
        lem_tokens.append(lem.lemmatize(word, pos = pos_t) + ' ' + pos_t)
    print('lem')
    return lem_tokens

def preprocess_data(file):
    data = open(file, encoding = 'latin1')
    lines = data.readlines()
    # Convert to lower case
    lines = [line.lower() for line in lines]
    print('Converted to lower case')
    # Split the message and the tag
    separated_lines = []
    for line in lines:
        comma = line.rfind(',')
        message = line[:comma]
        tag = line[comma:]
        separated_lines.append([message.split(),is_spam(tag)])
    print('Separated messages and tags')
    # POS tagging to every message
    t2 = create_tagger()
    tagged_lines = [[tagger(line[0],t2),line[1]] for line in separated_lines]
    print('POS tagging done!')
    # Lemmatize lines
    lem = WordNetLemmatizer()
    lem_lines = [[lemmatizer(line[0],lem),line[1]] for line in tagged_lines]
    print('Lemmatized')
    # Create the vocabulary
    aux = []
    for [l,t] in lem_lines:
        aux += l
    vocabulary = sorted(set(aux))
    print(lem_lines)
    print(len(lem_lines))
    return lem_lines, vocabulary

def is_spam(status):
    return 1 if status == 'spam' else 0

def vectorize(documents, vocabulary):
    vectors = []
    for [document,tag] in documents:
        i = 0
        vector = [0 for i in range(len(vocabulary))]
        vector.append(1) # This is for the independent parameter
        for word in vocabulary:
            vector[i] += document.count(word) / len(vocabulary)
            i += 1
        vectors.append([vector.copy(),tag])
    return vectors

def split_data(size, messages):
    messages_train, messages_test = [], []
    indices = [i for i in range(len(messages))]
    random.shuffle(indices)
    for i in indices[:size]:
        messages_train.append(messages[i])
    for i in indices[size:]:
        messages_test.append(messages[i])
    return messages_train, messages_test

def convert_vectors(messages_train, messages_test):
    X_train, y_train, X_test, y_test = [], [], [], []
    for [x,y] in messages_train:
        X_train.append(np.array(x))
        y_train.append(y)
    for [x,y] in messages_test:
        X_test.append(np.array(x))
        y_test.append(y)
    X_train = np.array(X_train)
    y_train = np.array([y_train])
    X_test = np.array(X_test)
    y_test = np.array([y_test])
    return X_train, y_train, X_test, y_test

def main():
    # Preprocessing data
    messages, vocabulary = preprocess_data('SMS_Spam_Corpus_big.txt')
    print('Data preprocessed')
    # Vectorize messages
    vectors = vectorize(messages,vocabulary)
    print('Vectors done!')
    # Split messages in a train and test set
    size = int(len(messages) * 0.7)
    messages_train, messages_test = split_data(size, vectors)
    print('Split done!')
    # Convert to numpy matrix and vector
    X_train, y_train, X_test, y_test = convert_vectors(messages_train, messages_test)
    save_structure(X_train,'X_train')
    save_structure(y_train,'y_train')
    save_structure(X_test,'X_test')
    save_structure(y_test,'y_test')
    print('Convertion done!')
    #Y = np.array([Y.values])
    lr = LogisticRegression(len(vocabulary+1),X_train,y_train)
    lr.train(30000)
    save_structure('lr.pkl')
    lr.test(X_test,messages_test)
main()
