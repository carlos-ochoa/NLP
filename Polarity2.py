import re
import os
import nltk
import numpy as np
from pickle import load,dump
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import xml.etree.ElementTree as ET
from prettytable import PrettyTable

LEM_DICT = '../Files/Dictionaries/lem_dict.pkl'

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

def write_dictionary(dict,filename):
    file = open(filename,'a')
    file.write('\n\n')
    for k,v in dict.items():
        file.write("%s %s\n" % (k,v))
    file.close()

def write_file(list,filename):
    file = open(filename,'a')
    file.write('\n\n')
    for (k,v) in list:
        file.write("%s %s\n" % (k,v))
    file.close()

def load_polarity_vocabulary():
    vocabulary = {}
    path = 'polarity/senticon.es.xml'
    tree = ET.parse(path)
    root = tree.getroot()
    for lemma in root.iter('lemma'):
        #vocabulary[(lemma.text + lemma.attrib['pos']).strip()] = float(lemma.attrib['pol'])
        vocabulary[(lemma.text).strip()] = float(lemma.attrib['pol'])
    return vocabulary

def load_polarity_vocabulary2():
    vocabulary = {}
    name = 'fullStrengthLexicon.txt'
    with open(name,'r',encoding = 'latin1') as f:
        lines = f.readlines()
        for line in lines:
            vocabulary[line.split()[0].lower()] = line.split()[-1]
    return vocabulary

def lematize_tokens(tokens,lemmas = None):
    lem_tokens = []
    if lemmas == None:
        lemmas = load_structure(LEM_DICT)
    for token in tokens:
        new_token = lemmas[token] if token in lemmas else token
        lem_tokens.append(new_token)
    return lem_tokens

def load_reviews():
    reviews,r = [],''
    # List of filenames to read
    path = 'corpusCriticasCine/'
    stopw = stopwords.words('spanish')
    PATTERN = r'[0-9]|¿|\-|—|\.+|\?|\\n|&|\*|%|@|(|)|~|\)|!|¡|\(|\\|\+|º|¦|\/|:|"|#|,|\$|`|;|_|{|\[|\]|\'+'
    files = [f for f in os.listdir(path) if f.endswith('.review.pos')]
    for name in files:
        with open(path+name,'r',encoding = 'latin1') as f:
            lines = f.readlines()
            for line in lines:
                # Extract the second token of the line
                if len(line) > 1:
                    word = line.split()[1]
                    # Only if the word is not a stopword and cleaning special characters
                    if word not in stopw and len(re.sub(PATTERN,r'',word)) > 0:
                        r += ',' + word + ' ' + line.split()[2][0].lower()
        reviews.append(r.split(','))
        r = ''
    return reviews

def get_ranks():
    ranks = []
    path = 'corpusCriticasCine/'
    files = [f for f in os.listdir(path) if f.endswith('.xml')]
    #files = [path+str(i)+'.xml' for i in range(2,4381) if i not in files_not_found]
    for name in files:
        with open(path+name,'r',encoding = 'latin1') as f:
            lines = f.readlines()
            # Take the first line where is the rank
            rank = int(lines[0][lines[0].rfind('rank') + 6])
        ranks.append(rank)
    return ranks

def load_computer_reviews():
    pos_words, neg_words = 0,0
    path = 'ordenadores/'
    reviews_yes,reviews_no,r,R = [],[],'',[]
    lemmas = load_structure(LEM_DICT)
    # List of filenames to read
    stopw = stopwords.words('spanish')
    # Debo agregar que elimine espacios
    PATTERN = r'[0-9]| |¿|\-|—|\.+|\?|\\n|&|\*|%|@|(|)|~|\)|!|¡|\(|\\|\+|º|¦|\/|:|"|#|,|\$|`|;|_|{|\[|\]|\'+'
    files = [f for f in os.listdir(path) if f.endswith('.txt')]
    for name in files:
        with open(path+name,'r',encoding = 'latin1') as f:
            lines = f.read()
            sentences = sent_tokenize(lines)
            for sent in sentences:
                # Extract the second token of the line
                if len(sent) > 1:
                    words = sent.split()
                    # Only if the word is not a stopword and cleaning special characters
                    for word in words:
                        word = re.sub(PATTERN,r'',word.lower())
                        if len(word) > 0 and word.lower() not in stopw:
                            # Lemmatize word
                            word = lemmas[word.lower()] if word.lower() in lemmas else word.lower()
                            R.append(word.lower())
                            r += ' ' + word.lower() + ' '
                if name.startswith('yes'):
                    reviews_yes.append(r.strip())
                    pos_words += len(r.split())
                else:
                    reviews_no.append(r.strip())
                    neg_words += len(r.split())
                r = ''
    return reviews_yes,reviews_no,R,pos_words,neg_words

def compute_ngrams(sequence,n):
    return list(zip(*[sequence[index:] for index in range(n)]))

def count_ngrams(ngrams):
    count = 0
    used = []
    ngram_register = []
    for gram in ngrams:
        if gram[0] not in used:
            for i in range(len(ngrams)):
                if gram[0] == ngrams[i][0]:
                    count += 1
            ngram_register.append((gram[0],count))
            used.append(gram[0])
        count = 0
    return sorted(ngram_register,key = lambda x : x[1], reverse = True)

def get_topic_polarities(reviews,topics,polarities,pos_words,neg_words):
    topic_pol = {topic:[] for topic in topics}
    top_five_pos_topic,top_five_neg_topic = {},{}
    top_five_pos,top_five_neg = {pol_word:0 for pol_word in polarities.keys()},{pol_word:0 for pol_word in polarities.keys()}
    for topic in topics:
        topic_found = 0
        topic_pos_polarity, topic_neg_polarity = 0,0
        top_five_pos,top_five_neg = {pol_word:0 for pol_word in polarities.keys()},{pol_word:0 for pol_word in polarities.keys()}
        for sentence in reviews:
            if topic in sentence:
                # Compute the sentence so that we can obtain its polarity
                for word in sentence.split():
                    if word in list(polarities.keys()):
                        if polarities[word] == 'pos':
                            topic_pos_polarity += 1
                            top_five_pos[word] += 1
                        else:
                            topic_neg_polarity += 1
                            top_five_neg[word] += 1
        if topic_pos_polarity > topic_neg_polarity:
            polarity = 'positive'
        elif topic_pos_polarity == topic_neg_polarity:
            polarity = 'neutral'
        else:
            polarity = 'negative'
        topic_pol[topic] = polarity
        top_five_pos = {k: round(v / pos_words,4) for k, v in sorted(top_five_pos.items(), key = lambda item: item[1],reverse = True)}
        top_five_neg = {k: round(v / neg_words,4) for k, v in sorted(top_five_neg.items(), key = lambda item: item[1],reverse = True)}
        top_five_pos_topic[topic] = list(top_five_pos.items())[:5].copy()
        top_five_neg_topic[topic] = list(top_five_neg.items())[:5].copy()
    return topic_pol,top_five_pos_topic,top_five_neg_topic

def most_frequent_words(topics,reviews,w_number):
    words,words_per_topic = {},{}
    for topic in topics:
        words = {}
        for sentence in reviews:
            if topic in sentence:
                for word in sentence.split():
                    if word not in words.keys():
                        words[word] = 1
                    else:
                        words[word] += 1
        words = {k: round(v / w_number,4) for k, v in sorted(words.items(), key = lambda item: item[1],reverse = True)}
        words_per_topic[topic] = list(words.items())[:5].copy()
    return words_per_topic

def get_normalized_polarities(reviews,ranks,polarities):
    i,word_found = 0,0
    rank_pol = {1:[],2:[],3:[],4:[],5:[]}
    for review in reviews:
        review_polarity = 0
        word_found = 0
        for word in review:
            if word in list(polarities.keys()):
                review_polarity += polarities[word]
                word_found += 1
        if word_found != 0:
            review_polarity = review_polarity / word_found
        else:
            review_polarity = 0
        rank_pol[ranks[i]].append(review_polarity)
        i += 1
    for k,v in rank_pol.items():
        rank_pol[k] = np.mean(np.array(v))
    return rank_pol

def main():
    topics = ['ordenador','disco','windows','mac','memoria','diseño','ram']
    final_reviews = {k:[] for k in topics}
    #reviews = load_reviews()
    #save_structure(reviews,'reviews.pkl')
    reviews_yes,reviews_no,r,pos_words,neg_words = load_computer_reviews()
    print('Reviews loaded!')
    polarities = load_polarity_vocabulary2()
    #print(polarities)
    print('Polarity loaded!')
    ngrams = compute_ngrams(r,1)
    print('Unigrams calculated')
    ngrams = count_ngrams(ngrams)
    #print(ngrams)
    #write_file(ngrams,'unigrams.txt')
    #write_dictionary(rank_pol,'rank_pol.txt')
    words_yes = most_frequent_words(topics,reviews_yes,pos_words+neg_words)
    words_no = most_frequent_words(topics,reviews_no,pos_words+neg_words)
    print('Words counted')
    topic_pol_yes,top_five_pos_yes,top_five_neg_yes = get_topic_polarities(reviews_yes,topics,polarities,pos_words,neg_words)
    topic_pol_no,top_five_pos_no,top_five_neg_no = get_topic_polarities(reviews_no,topics,polarities,pos_words,neg_words)
    print('Polarity obtained')
    #x = PrettyTable()
    #x.field_names = ['Categoría','Pol. en yes','Pol. en no','Pal. mas probables yes','Pal. mas probables no','Pal pos. mas prob. yes','Pal neg. mas prob. yes','Pal pos. mas prob. no','Pal neg. mas prob. no']
    for k,v in final_reviews.items():
        print('Categoría: ',k)
        print('Pol. en yes: ',topic_pol_yes[k])
        print('Pol. en no: ',topic_pol_no[k])
        print('Pal. mas probables yes: ',words_yes[k])
        print('Pal. mas probables no: ',words_no[k])
        print('Pal pos. mas prob. yes: ',top_five_pos_yes[k])
        print('Pal neg. mas prob. yes: ',top_five_neg_yes[k])
        print('Pal pos. mas prob. no: ',top_five_pos_no[k])
        print('Pal neg. mas prob. no: ',top_five_neg_no[k])
        print('\n')
    #    x.add_row(v)
        v.append(final_reviews[k])
        v.append(topic_pol_yes[k])
        v.append(topic_pol_no[k])
        v.append(words_yes[k])
        v.append(words_no[k])
        v.append(top_five_pos_yes[k])
        v.append(top_five_neg_yes[k])
        v.append(top_five_pos_no[k])
        v.append(top_five_neg_no[k])
    #    x.add_row(v)
    #print(x)
if __name__ == '__main__':
    main()
