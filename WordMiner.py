import nltk
import numpy as np
from pickle import dump,load

FILE_PIK_COS = 'cosines_l.pkl'
FILE_PIK_VECT = 'vectors_l.pkl'
FILE_COS = 'cosines_l.txt'

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

def write_file(info,filename):
    file = open(filename,'w')
    for line in info:
        file.write("%s\n" % line)
    file.close()

def write_dictionary(dict,filename):
    file = open(filename,'w')
    for k,v in dict.items():
        file.write("%s %s\n" % (k,v))
    file.close()

def read_file(filename):
    file = open(filename,'r')
    text = file.read()
    return text

# Load an html file
def load_html_file(file):
    f = open(file,encoding = 'utf-8')
    text = f.read()
    f.close()
    return text

# Vectorization
def vectorize_tokens(contexts, vocabulary):
    vectors = {}
    vector = []
    for word in vocabulary:
        context = contexts[word]
        for w in vocabulary:
            frecuency = context.count(w)
            vector.append(frecuency)
        vectors[word] = vector.copy()
        vector.clear()
    save_structure(vectors,FILE_PIK_VECT)
    return vectors

def vectorize_tokens_c(contexts,vocabulary):
    vectors = {}
    for word in vocabulary:
        context = contexts[word]
        vector = [0 for i in range(len(vocabulary))]
        for w in context:
            index = vocabulary.index(w)
            vector[index] += 1
        vectors[word] = vector
    save_structure(vectors,FILE_PIK_VECT)
    return vectors

# Cosine calculation
def calculate_cosines(main_word,vectors,use_tags = False):
    cosines = {}
    vect_A = np.array(vectors[main_word])
    if use_tags:
        vectors = filter_vectors(main_word,vectors)
    for word,v in vectors.items():
        vect_B = np.array(vectors[word])
        cosine = (np.dot(vect_A,vect_B)) / np.multiply(np.sqrt(np.sum(vect_A**2)),np.sqrt(np.sum(vect_B**2)))
        cosines[word] = cosine
    cosines = {k:v for k,v in sorted(cosines.items(), key = lambda item: item[1], reverse = True)}
    save_structure(cosines,FILE_PIK_COS)
    write_dictionary(cosines,FILE_COS)
    return cosines

# Auxiliar filter function to accept just the same POS tokens
def filter_vectors(main_word,vectors):
    filtered_vectors = {}
    tag = main_word.split()[1]
    for word,vect in vectors.items():
        if word.split()[1] == tag:
            filtered_vectors[word] = vect
    return filtered_vectors
