from nltk.book import *
#from __future__ import division

print("\n\nCOUNTING VOCABULARY\n\n")
print("Length of the text: " + str(text3) + " is " + str(len(text3)) + " tokens")
print("\n\nUnique items in the text: " + str(sorted(set(text3))))

print("\n\nCalculating lexical richness of Genesis\n")
print(str(len(text3) / len(set(text3))))

print("\n\nCalculating the ocurrence of the word 'God' in " + str(text3))
print(str(text3.count("God")))
res = text3.count("God") / len(text3) * 100
print("The word 'God' is in the " + str(res) + "% of the text")
