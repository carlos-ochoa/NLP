import nltk
from nltk.corpus import brown
from operator import itemgetter

counts = nltk.defaultdict(int)
for (word,tag) in brown.tagged_words(categories = 'news'):
    counts[tag] += 1

l = sorted(counts.items(), key = itemgetter(1), reverse = True)

# Inverting a dictionary
pos = {'colorless':'ADJ','ideas':'N','sleep':'V','furiously':'ADV'}
pos2 = dict((value,key) for (key,value) in pos.items())
