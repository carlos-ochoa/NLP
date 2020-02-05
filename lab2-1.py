import nltk
from nltk.corpus import PlaintextCorpusReader

corpus_root = 'Corpus'
wordlists = PlaintextCorpusReader(corpus_root,'.*')
print(wordlists.fileids())
words = list(wordlists.words('e961024.htm'))
print(words)
