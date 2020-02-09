import nltk
import re
import string
from bs4 import BeautifulSoup
from nltk.corpus import PlaintextCorpusReader

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
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('',token) for token in tokens])
    filtered_tokens = [t for t in list(filtered_tokens) if len(re.sub(r'[0-9]+.*',r'',t)) > 0]
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
    return filtered_tokens

# Get vocabulary of the text
def get_vocabulary(tokens):
    vocabulary = sorted(set(tokens))
    return vocabulary
