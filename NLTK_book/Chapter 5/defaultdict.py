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
print(pos2['N'])
pos.update({'cats':'N','scratch':'V','peacefully':'ADV','old','ADJ'})
pos2 = nltk.defaultdict(list)
for key,value in pos.items():
    pos2[value].append(key)
