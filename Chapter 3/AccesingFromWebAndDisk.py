import nltk
from urllib import request

url = "http://www.gutenberg.org/files/2554/2554-0.txt"
raw = request.urlopen(url).read().decode('utf-8')
print(type(raw))
print(len(raw))
print(raw[:75])

print('\nTokenization\n')
tokens = nltk.word_tokenize(raw)
print(tokens[:20])

text = nltk.Text(tokens)
print(text.collocations())
