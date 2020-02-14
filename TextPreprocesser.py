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
    #PATTERN =
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('',token) for token in tokens])
    filtered_tokens = [t for t in list(filtered_tokens) if len(re.sub(r'[0-9]+.*|\-\+',r'',t)) > 0]
    #filtered_tokens = [t for t in filtered_tokens if ]
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
    half = limit - 1
    for word in tokens:
        # Only at the beginning of the algorithm
        if i == 0:
            memo_context = tokens[:windowSize+1]
            context = memo_context[i+1:limit]
            contexts[word] = context
            i += 1
        else:
            if i <= limit - 2:
                context = memo_context[:i] + memo_context[i+1:i+limit]
            elif i < len(tokens) - limit:
                context = memo_context[:half] + memo_context[half+1:]
                memo_context.pop(0)
                memo_context.append(tokens[i+limit])
            else:
                context = memo_context[i-limit:i] + memo_context[i:]
            # Verify if the word already exists in the dictionary
            if word in contexts:
                aux_context = contexts[word] + context
                contexts[word] = aux_context
            else:
                contexts[word] = context
            i += 1
    return contexts
