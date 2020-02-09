import re, collections

def tokens(text):
    # Get all the words of the corpus
    return re.findall('[a-z]+',text.lower())

f = open('big.txt')
WORDS = tokens(f.read())
WORD_COUNTS = collections.Counter(WORDS)

print(WORD_COUNTS.most_common(10))

def edits0(word):
    # Return all the string that are zero edits away from the word
    return {word}

def edits1(word):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    def splits(word):
        # Return a list of all possible (first,rest) pairs that the input word
        # is made of
        return [(word[:i],word[i:]) for i in range(len(word)+1)]
    pairs = splits(word)
    deletes = [a+b[1:] for (a,b) in pairs if b]
    transposes = [a+b[1] + b[0] + b[2:] for (a,b) in pairs if len(b) > 1]
    replaces = [a + c + b[1:] for (a,b) in pairs for c in alphabet if b]
    inserts = [a + c +b for (a,b) in pairs for c in alphabet]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    return {e2 for e1 in edits1(word) for e2 in edits1(e1)}

def known(words):
    # Return the subset of words that are actually in our WORD_COUNTS dictionary
    return {w for w in words if w in WORD_COUNTS}

def correct(word):
    # Priority is for edit distance 0, then 1, then 2
    # else defaults to the input word itself
    candidates = (known(edits0(word)) or
                    known(edits1(word)) or
                    known(edits2(word)) or
                    [word]
                )
    return max(candidates, key = WORD_COUNTS.get)

word = 'fianlly'
