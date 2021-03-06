import nltk
import re
import string
import numpy as np
from pickle import dump,load
from bs4 import BeautifulSoup
from nltk.corpus import PlaintextCorpusReader
from nltk.stem import SnowballStemmer

FILE_PIK_VOC = '../Files/Vocabularies/vocabulary_s2.pkl'
FILE_PIK_TOK = '../Files/Tokens/tokens_s2.pkl'
FILE_PIK_CONT = '../Files/Contexts/contexts_s2.pkl'
FILE_PIK_FILTERED = '../Files/Tokens/tokens_f.pkl'
FILE_PIK_TAGGER = '../Files/tagger.pkl'
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

def write_file(info,filename):
    file = open(filename,'w')
    for line in info:
        file.write("%s\n" % line)
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

# Clean html text
def clean_html_text(text):
    soup = BeautifulSoup(text,'lxml')
    text = soup.get_text()
    return text

# Tokenize words
def tokenize_text_nltk(text):
    tokens = nltk.word_tokenize(text)
    return tokens

def tokenize_text(text):
    tokens = text.split()
    tokens = [t.lower() for t in tokens if len(re.sub(r'[0-9]',r'',t)) > 0] # This line eliminates only numbers in the tokens set
    return tokens

# Remove characters after tokenization
def remove_characters_after_tokenization(tokens):
    PATTERN = r'[0-9]|¿|\-|—|\.+|\?|&|\*|%|@|(|)|~|\)|!|¡|\(|\\|\+|º|¦|\/|:|"|#|,|\$|`|;|_|{|\[|\]|\'+'
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('',token) for token in tokens])
    #filtered_tokens = [t for t in list(filtered_tokens) if len(re.sub(r'[0-9]+.*|¿',r'',t)) > 0]
    filtered_tokens = [re.sub(PATTERN,r'',t) for t in tokens if len(re.sub(PATTERN,r'',t)) > 0]
    return list(filtered_tokens)

# Remove special characters before tokenization
def remove_characters_before_tokenization(sentence,keep_apostrophes = False):
    sentence = sentence.strip()
    if keep_apostrophes:
        PATTERN = r'[?|$|&|*|%|@|(|)|~]' # Add more characters here to remove them
        filtered_sentence = re.sub(PATTERN,r'',sentence)
    else:
        PATTERN = r'[^a-zA-Z0-9 ]' # Only extract alpha numeric characters
        filtered_sentence = re.sub(PATTERN,r'',sentence)
    return filtered_sentence

# Remove stopwords
def remove_stopwords(tokens):
    stopword_list = nltk.corpus.stopwords.words('spanish')
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    save_structure(filtered_tokens,FILE_PIK_FILTERED)
    return filtered_tokens

# Get vocabulary of the text
def get_vocabulary(tokens):
    vocabulary = sorted(set(tokens))
    save_structure(vocabulary,FILE_PIK_VOC)
    return vocabulary

# Get context (Kolesnikova's implementation)
def get_context(tokens, windowSize = 8):
    contexts = {}
    for w in tokens:
        context2 = []
        for i in range(len(tokens)):
            if tokens[i] == w: # w is the current word analyzed
                for j in range(i - int(windowSize / 2), i): # left context
                    if j >= 0:
                        context2.append(tokens[j])
                try:
                    for j in range(i+1, i+(int(windowSize/2)+1)): # right context
                        context2.append(tokens[j])
                except IndexError:
                    pass
        contexts[w] = context2
    return contexts

# Get context (Charly's implementation)
def get_context_c(tokens, windowSize = 8):
    contexts = {}
    context = []
    aux_context = []
    memo_context = []
    i = 0
    limit = int(windowSize / 2) + 1
    j = limit
    half = limit - 1
    for word in tokens:
        # Only at the beginning of the algorithm
        if i == 0:
            memo_context = tokens[:windowSize+1]
            context = memo_context[i+1:limit]
            contexts[word] = context
            i += 1
        else:
            if i < limit - 2:
                context = memo_context[:i] + memo_context[i+1:i+limit]
            elif i < len(tokens) - limit:
                context = memo_context[:half] + memo_context[half+1:]
                memo_context.pop(0)
                memo_context.append(tokens[i+limit])
            else:
                if j != half*2+1:
                    # We make a copy of memo_context
                    context = memo_context[j-half:j] + memo_context[j+1:]
                    j += 1
            # Verify if the word already exists in the dictionary
            if word in contexts:
                aux_context = contexts[word] + context
                contexts[word] = aux_context
            else:
                contexts[word] = context
            i += 1
    save_structure(contexts,FILE_PIK_CONT)
    return contexts

# Tag tokens
def tag_tokens(tokens, tagger = None):
    tagged_tokens = []
    # Problema con los continuos accesos a este archivo
    if tagger == None:
        tagger = load_structure(FILE_PIK_TAGGER)
    tagged_tokens_tuples = tagger.tag(tokens)
    for (token,tag) in tagged_tokens_tuples:
        tagged_tokens.append(token + " " + tag[0].lower())
    return tagged_tokens

# Lematization functions
# This implementation is for a python dictionary, nltk dictionary will be implemented later
def lematize_tokens(tokens,lemmas = None):
    lem_tokens = []
    # El problema está en los continuos accesos a este archivo
    if lemmas == None:
        lemmas = load_structure(LEM_DICT)
    for token in tokens:
        new_token = lemmas[token] if token in lemmas else token
        lem_tokens.append(new_token)
    return lem_tokens

# Stemming
def stemming_tokens(tokens):
    stem_tokens = []
    stems = load_structure('stem_dict.pkl')
    for token in tokens:
        new_token = stems[token] if token in stems else token
        '''if token in lemmas:
            new_token = lemmas[token]
        else:
            new_token ='''
        stem_tokens.append(new_token)
    return stem_tokens

def stemming_tokens_ss(tokens):
    ss = SnowballStemmer('spanish')
    stems = []
    for token in tokens:
        stems.append(ss.stem(token))
    return stems
