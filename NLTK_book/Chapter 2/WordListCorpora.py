import nltk
from nltk.corpus import swadesh

puzzle_letters = nltk.FreqDist('egivrvonl')
obligatory = 'r'
wordlist = nltk.corpus.words.words()
words = [w for w in wordlist if len(w) >= 6 and obligatory in w and nltk.FreqDist(w) <= puzzle_letters]
print(words)

print("\n\nGenerating a simple translator")
es2en = swadesh.entries(['es','en'])
translate = dict(es2en)
de2en = swadesh.entries(['de','en'])
translate.update(de2en)
print(translate['Hund'])
print(translate['perro'])
