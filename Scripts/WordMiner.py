import nltk
import math
import numpy as np
from pickle import dump,load

FILE_PIK_COS = '../Files/Cosines/cosines_s_tf_2.pkl'
FILE_PIK_VECT = '../Files/Vectors/vectors_s_tf_2.pkl'
FILE_COS = '../Files/Cosines/cosines_s_tf_2.txt'
FILE_PIK_SYN = '../Files/Cosines/cosines_syn_tf_2.pkl'
FILE_SYN = '../Files/Cosines/cosines_syn_tf_2.txt'
FILE_PIK_ENT = '../Files/Cosines/cosines_syn.pkl'
FILE_ENT = '../Files/Cosines/entropies_syn.txt'

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

# Using the other BM25 formula
def vectorize_tf_idf2(contexts,vocabulary,k = 1.2, b = 0.75):
    final_vectors = {}
    vectors_tf = {}
    vectors_tf_final = {}
    raw_vectors = vectorize_tokens_c(contexts,vocabulary)
    # Calculate the average length of every document
    avg = get_avg_len(raw_vectors)
    # Apply the tf funtion : BM25 transformation
    for word,vector in raw_vectors.items():
        v_tf = ((k+1) * np.array(vector)) / (np.array(vector) + k*(1 - b * (np.sum(np.array(vector))) / avg))
        vectors_tf[word] = v_tf
    # Normalization for BM25
    sum_vec_tf = np.array([0.0 for v in vectors_tf])
    for word,vector in vectors_tf.items():
        sum_vec_tf += vector
    for word,vector in vectors_tf.items():
        vectors_tf_final[word] = vector / sum_vec_tf
    # Now apply the IDF step
    vector_doc_frec = calculate_word_frequency(contexts,vocabulary)
    vector_idf = get_idf_vector(vocabulary,vector_doc_frec)
    # Final step, we get the last vectors
    for word,vector in vectors_tf_final.items():
        v = np.multiply(vector,np.array(vector_idf))
        final_vectors[word] = list(v)
    return final_vectors

def compare_words(main_word, vectors, contexts, vocabulary):
    results = {}
    main_vector = vectors[main_word]
    vector_doc_frec = calculate_word_frequency(contexts,vocabulary)
    vector_idf = get_idf_vector(vocabulary,vector_doc_frec)
    for word,vector in vectors.items():
        #comp_measure = np.dot((np.array(main_vector) * np.array(vector)),np.array(vector_idf))
        comp_measure = np.dot(np.array(main_vector),np.array(vector))
        results[word] = comp_measure
    sorted_results = {k:v for k,v in sorted(results.items(), key = lambda item: item[1], reverse = True)}
    save_structure(sorted_results,FILE_PIK_SYN)
    write_dictionary(sorted_results,FILE_SYN)
    print(sorted_results['dólar n'])
    print(vector_idf)
    return sorted_results

def get_avg_len(vectors):
    sum = 0
    avg_len = 0
    for w,v in vectors.items():
        sum += np.sum(np.array(v))
    avg_len = sum / len(vectors)
    return avg_len

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

# Functions for syntagmatic relation and word prediction tasks
def calculate_probabilities(sentences,word1,word2):
    sum_w1, sum_w2, sum_w1_w2 = 0,0,0
    prob_w1, prob_w2, prob_w1_w2 = 0 ,0, 0
    for sentence in sentences:
        if word1 in sentence:
            sum_w1 += 1
        if word2 in sentence:
            sum_w2 += 1
        if word1 in sentence and word2 in sentence:
            sum_w1_w2 += 1
    prob_w1 = sum_w1 / len(sentences)
    prob_w2 = sum_w2 / len(sentences)
    prob_w1_w2 = sum_w1_w2 / len(sentences)
    return [prob_w1,prob_w2,prob_w1_w2]

def calculate_entropy(sentences,word1,word2):
    probabilities = calculate_probabilities(sentences,word1,word2)
    p_w1_0_w2_1 = probabilities[1] - probabilities[2]
    p_w2_0 = 1 - probabilities[1]
    p_w1_0 = 1 - probabilities[0]
    p_w1_0_w2_0 = p_w1_0 - p_w1_0_w2_1
    p_w1_1_w2_0 = p_w2_0 - p_w1_0_w2_0
    try:
        prob1 = math.log((p_w1_0_w2_0 / p_w2_0),2)
    except:
        prob1 = 0
    try:
        prob2 = math.log((p_w1_1_w2_0 / p_w2_0),2)
    except:
        prob2 = 0
    try:
        prob3 = math.log((p_w1_0_w2_1 / probabilities[1]),2)
    except:
        prob3 = 0
    try:
        prob4 = math.log((probabilities[2] / probabilities[1]),2)
    except:
        prob4 = 0
    H = -1 * (p_w1_0_w2_0 * prob1 + p_w1_1_w2_0 * prob2 \
                + p_w1_0_w2_1 * prob3 + probabilities[2] * prob4)
    return H

def discover_syntagmatic_relations(main_word,sentences,vocabulary, threshold = 0.0):
    entropies = {}
    for word in vocabulary:
        entropy = calculate_entropy(sentences,main_word,word)
        entropies[word] = entropy
    entropies = {k:v for k,v in sorted(entropies.items(), key = lambda item : item[1] , reverse = False) if v <= threshold}
    save_structure(entropies,FILE_PIK_ENT)
    write_dictionary(entropies,FILE_ENT)
    return entropies
