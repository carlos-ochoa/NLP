from nltk.corpus import PlaintextCorpusReader

corpus_root = '../Corpus'
wordlists = PlaintextCorpusReader(corpus_root,".*")
print(wordlists.fileids())
print(wordlists.words('Saludos.txt'))
print(wordlists.sents('Saludos.txt'))
