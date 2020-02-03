from nltk.book import *

def get_mc_words(mcw):
    ncw = []
    for tuple in mcw:
        ncw.append(tuple[0])
    return ncw

saying = ["After","all","is","said","and","done","more","is","said","than","done"]
tokens = sorted(set(saying))
print(tokens)
print(tokens[-2:])

print("\n\nCalculating the frequency distribution of the 50 most frequent words in text "+ str(text1))
fdist1 = FreqDist(text1)
print(str(fdist1))
print(fdist1.most_common(50))
print(type(fdist1.most_common(50)))
newList = get_mc_words(fdist1.most_common(50))
print(newList)

print("\n\nCumulative frequency of the words\n")
fdist1.plot(50,cumulative=True)

print("\n\nChecking the hapaxes (words that occur only once) in text " + str(text1))
print(fdist1.hapaxes())
