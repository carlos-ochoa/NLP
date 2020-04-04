import numpy as np
import matplotlib.pyplot as plt
from pickle import load, dump
import nltk
import random
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
        #print(self.Y)
        uno, cero = 0,0
        i = 0
        for e in self.Y:
            if e[0] == 0:
                cero += 1
            elif e[0] == 1:
                uno += 1
        print(cero)
        print(uno)
        print(params)
        #print(self.Y)
        self.l = 0
        self.total_loss = []
        self.w = np.array([np.random.rand(params)])

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
            p.append(1 if v >= 0.5 else 0)
        p = np.array([np.array(p)])
        return p.T

    def loss(self,h):
        epsilon = 1e-5
        return -1 * np.mean(self.Y * np.log(h + epsilon) + (1 - self.Y) * np.log(1 - h + epsilon))

    def gradient_descent(self, alpha = 0.1):
        error = self.sigmoid(self.hypothesis()) - self.Y
        #print(error)
        #print(self.X.shape)
        m = self.Y.size
        dw = []
        i = 0
        #dw = np.dot(error.T,self.X) * (1/m)
        for x in self.X:
            dw.append(error[i] * x)
            i += 1
        dw = np.array(dw)
        dw = np.sum(dw, axis = 0) / len(self.X)
        #print(dw.shape)
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
            if i % 50 == 0:
                print('Current loss ', self.l, 'at iter: ', i)
        save_structure(self.w, 'w.pkl')
        plt.plot(range(iter), self.total_loss)
        plt.show()

    def test(self,X,messages,y):
        i = 0
        self.X = X
        correctos = 0
        mess, is_spam = [], []
        predictions = self.predict(self.hypothesis())
        print(predictions)
        for x in X:
            mess.append(messages[i])
            is_spam.append(predictions[i][0])
            if y[0][i] == predictions[i][0]:
                correctos += 1
            print('Expected ',y[0][i],' Predicted : ', predictions[i][0])
            i += 1
        print(correctos)
        print(len(y[0]))
        #output.to_csv('output.csv', index = False)

def create_tagger():
    chat_tags = nps_chat.tagged_posts()
    t0 = nltk.DefaultTagger('NN')
    t1 = nltk.UnigramTagger(chat_tags, backoff = t0)
    t2 = nltk.BigramTagger(chat_tags, backoff = t1)
    return t2

def tagger(tokens,t2):
    tags = t2.tag(tokens)
    return tags

def lemmatizer(tokens,lem):
    lem_tokens = []
    for (word,pos_t) in tokens:
        lem_tokens.append(lem.lemmatize(word) + ' ' + pos_t)
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
        # Tokenize
        separated_lines.append([nltk.word_tokenize(message),is_spam(tag)])
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
    return lem_lines, vocabulary

def is_spam(status):
    spam = 1 if status == ',spam\n' else 0
    return spam

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

def split_data(perc, messages):
    messages_train, messages_test = [], []
    indices = []
    indices_spam = [i for i in range(len(messages)) if messages[i][1] == 1]
    indices_ham = [i for i in range(len(messages)) if messages[i][1] == 0]
    print(len(indices_spam))
    print(len(indices_ham))
    random.shuffle(indices_spam)
    random.shuffle(indices_ham)
    size = int(len(indices_spam) * perc)
    for i in indices_spam[:size]:
        messages_train.append(messages[i])
    for i in indices_spam[size:]:
        messages_test.append(messages[i])
    indices += indices_spam[:size]
    size = int(len(indices_ham) * perc)
    indices += indices_ham[:size]
    c = 0
    for i in indices_ham[:size]:
        if c < 224:
            messages_train.append(messages[i])
        c += 1
    for i in indices_ham[size:]:
        messages_test.append(messages[i])
    return messages_train, messages_test, indices

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
    '''messages, vocabulary = preprocess_data('SMS_Spam_Corpus_big.txt')
    print(len(vocabulary))
    print('Data preprocessed')
    # Vectorize messages
    vectors = vectorize(messages,vocabulary)
    print('Vectors done!')
    # Split messages in a train and test set
    messages_train, messages_test, indices = split_data(0.7, vectors)
    m_t = [messages[i][0] for i in indices]
    save_structure(m_t, 'm_t.pkl')
    #print(m_t)
    print('Split done!')
    # Convert to numpy matrix and vector
    X_train, y_train, X_test, y_test = convert_vectors(messages_train, messages_test)
    print(X_train.shape)
    print(X_test.shape)
    save_structure(X_train,'X_train.pkl')
    save_structure(y_train,'y_train.pkl')
    save_structure(X_test,'X_test.pkl')
    save_structure(y_test,'y_test.pkl')
    save_structure(vocabulary, 'vocabulary.pkl')
    print('Convertion done!')
    #vocabulary, X_train, y_train = load_structure('vocabulary.pkl'), load_structure('X_train.pkl'), load_structure('y_train.pkl')
    #Y = np.array([Y.values])
    lr = LogisticRegression(len(vocabulary)+1,X_train,y_train)
    lr.train(1001)
    save_structure(lr,'lr.pkl')'''
    lr = load_structure('lr.pkl')
    X_test , m_t, y_test = load_structure('X_test.pkl'), load_structure('m_t.pkl'), load_structure('y_test.pkl')
    lr.test(X_test,m_t,y_test)
main()
