import nltk
import math
import numpy as np
from pickle import dump,load

FILE_PIK_COS = '../Files/Cosines/cosines_s_tf.pkl'
FILE_PIK_VECT = '../Files/Vectors/vectors_s_tf.pkl'
FILE_COS = '../Files/Cosines/cosines_s_tf.txt'

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

# This is using normalized frequency
def vectorize_frec(contexts,vocabulary):
    raw_vectors = vectorize_tokens_c(contexts,vocabulary)
    normalized_vectors = {}
    for word,vector in raw_vectors.items():
        n_vector = np.array(vector) / np.sum(np.array(vector))
        normalized_vectors[word] = list(n_vector)
    return normalized_vectors

def vectorize_tf_idf(contexts,vocabulary,k = 1.2):
    final_vectors = {}
    vectors_tf = {}
    raw_vectors = vectorize_tokens_c(contexts,vocabulary)
    # Apply the tf funtion : BM25 transformation
    for word,vector in raw_vectors.items():
        v_tf = ((k+1) * np.array(vector)) / (np.array(vector) + k)
        vectors_tf[word] = v_tf
    # Now apply the IDF step
    vector_doc_frec = calculate_word_frequency(contexts,vocabulary)
    vector_idf = get_idf_vector(vocabulary,vector_doc_frec)
    # Final step, we get the last vectors
    for word,vector in vectors_tf.items():
        v = np.multiply(vector,np.array(vector_idf))
        final_vectors[word] = list(v)
    return final_vectors

# IDF function
def get_idf_vector(vocabulary,vector_doc_frec):
    vector_idf = []
    i = 0
    for word in vocabulary:
        # Calculate the IDF
        idf = math.log((len(vocabulary)+1) / vector_doc_frec[i])
        vector_idf.append(idf)
        i += 1
    return vector_idf

def calculate_word_frequency(contexts,vocabulary):
    vector_doc_frec = [0 for c in contexts]
    i = 0
    for word in vocabulary:
        for w,c in contexts.items():
            if word in c:
                vector_doc_frec[i] += 1
        i += 1
    return vector_doc_frec

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
