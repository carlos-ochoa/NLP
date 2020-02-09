import nltk
import re
import string
from pprint import pprint
from contractions import CONTRACTION_MAP
from nltk.corpus import wordnet

corpus = ["The brown fox wasn't that quick and he couldn't win the race","Hey that's a great deal! I just bought a phone for $199","@@You'll (learn) a **lot** in the book. Python is an amazing language!@@"]

def tokenize_text(text):
    sentences = nltk.sent_tokenize(text)
    word_tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
    return word_tokens

token_list = [tokenize_text(text) for text in corpus]

pprint(token_list)

def remove_characters_after_tokenization(tokens):
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('',token) for token in tokens])
    return list(filtered_tokens)

filtered_list = [filter(None,[remove_characters_after_tokenization(tokens) for tokens in sentence_tokens]) for sentence_tokens in token_list]

for f in filtered_list:
    for e in f:
        print(e)

def remove_characters_before_tokenization(sentence,keep_apostrophes = False):
    sentence = sentence.strip()
    if keep_apostrophes:
        PATTERN = r'[?|$|&|*|%|@|(|)|~]' # Add more characters here to remove them
        filtered_sentence = re.sub(PATTERN,r'',sentence)
    else:
        PATTERN = r'[^a-zA-Z0-9 ]' # Only extract alpha numeric characters
        filtered_sentence = re.sub(PATTERN,r'',sentence)
    return filtered_sentence

filtered_list2 = [remove_characters_before_tokenization(sentence) for sentence in corpus]

print(filtered_list2)

cleaned_corpus = [remove_characters_before_tokenization(sentence,keep_apostrophes = True) for sentence in corpus]

def expand_contractions(sentence,contraction_mapping):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), flags = re.IGNORECASE|re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_sentence = contractions_pattern.sub(expand_match, sentence)
    return expanded_sentence

expanded_corpus = [expand_contractions(sentence,CONTRACTION_MAP) for sentence in cleaned_corpus]

print(expanded_corpus)

def remove_stopwords(tokens):
    stopword_list = nltk.corpus.stopwords.words('english')
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    return filtered_tokens

expanded_corpus_tokens = [tokenize_text(text) for text in expanded_corpus]
filtered_list3 = [[remove_stopwords(tokens) for tokens in sentence_tokens] for sentence_tokens in expanded_corpus_tokens]

print(filtered_list3)

# Correcting repeated characters
old_word = 'finalllyyy'
repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
match_substitution = r'\1\2\3'
step = 1

while True:
    # check for semantically correct word
    if wordnet.synsets(old_word):
        print("Final correct word : " + old_word)
        break
    # remove one repeated character
    new_word = repeat_pattern.sub(match_substitution,old_word)
    if new_word != old_word:
        print('Step: {} Word: {}'.format(step,new_word))
        step += 1 # update step
        # update old word to last substituted state
        old_word = new_word
        continue
    else:
        print("Final word: " + new_word)
        break

def remove_repeated_characters(tokens):
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    match_substitution = r'\1\2\3'
    def replace(old_word):
        if wordnet.synsets(old_word):
            return old_word
        new_word = repeat_pattern.sub(match_substitution,old_word)
        return replace(new_word) if new_word != old_word else new_word

    correct_tokens = [replace(word) for word in tokens]
    return correct_tokens

sample_sentence = 'My school is realllllyyy amazzzinnnnggg'
sample_sentence_tokens = tokenize_text(sample_sentence)[0]
print(sample_sentence_tokens)

print(remove_repeated_characters(sample_sentence_tokens))
