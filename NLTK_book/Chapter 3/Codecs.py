import codecs
import nltk

path = nltk.data.find('corpora/unicode_samples/polish-lat2.txt')
f = codecs.open(path,encoding = 'latin2')

for line in f:
    line = line.strip()
    print(line.encode('unicode_escape'))
