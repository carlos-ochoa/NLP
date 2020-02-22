import re
import nltk

wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]
# ed$ means that we are looking for words that end with 'ed'
matched = [w for w in wordlist if re.search('..j..t..$',w)]
print(matched[:20])
