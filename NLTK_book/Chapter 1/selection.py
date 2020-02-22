from nltk.book import *

V = set(text6)
fdist = FreqDist(text6)
print(str(fdist.keys()))
long_words = [w for w in V if len(w) > 7 and fdist[w] > 3]
print(sorted(long_words))

print("\n\nBigrams and collocations in text " + str(text5))
bi = bigrams(["more","is","said","than","done"])
print(bi)
print(text5.collocations())

print("\n\nCounting Length of words and getting their frequency distribution\n")
leng = [len(w) for w in V]
fdist2 = FreqDist([len(w) for w in text6])
print(str(fdist2.items()))
print("\nFrecuencia " + str(fdist2.freq(1)))
fdist2.plot()
