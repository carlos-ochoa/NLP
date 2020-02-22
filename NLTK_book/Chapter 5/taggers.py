import nltk
from nltk.corpus import brown, treebank, cess_esp
from pickle import load,dump

text = nltk.word_tokenize("And now for something completely different")
print(nltk.pos_tag(text))

print('\nChecking the most common order of words in the Brown corpus\n')
brown_news_tagged = brown.tagged_words(categories = 'news', tagset = 'universal')
tag_fd = nltk.FreqDist(tag for (word,tag) in brown_news_tagged)
tag_fd.keys()
word_tag_pairs = nltk.bigrams(brown_news_tagged)
res = list(nltk.FreqDist(a[1] for (a,b) in word_tag_pairs if b[1] == "NOUN"))
print(res)

brown_sents = brown.sents(categories = 'news')

# Default tagger
print('\n\nDefault tagger\n')
tags = [tag for (word,tag) in brown.tagged_words(categories = 'news')]
print(nltk.FreqDist(tags).max())

raw = 'I do not like green eggs and ham, I do not like then Sam I am! He should be eating'
tokens = nltk.word_tokenize(raw)
default_tagger = nltk.DefaultTagger('NN')
tags = default_tagger.tag(tokens)
print(tags)

# Regular Expression Tagger
patterns = [
    (r'.*ing$','VBG'),
    (r'.*ed$','VBD'),
    (r'.*es$','VBZ'),
    (r'.*ould$','MD'),
    (r'.*\'s$','NN$'),
    (r'.*s$','NNS'),
    (r'^-?[0-9]+(.[0-9]+)?$','CD'),
    (r'.*','NN')
]

print('\n\nRegexp tagger\n')
regexp_tagger = nltk.RegexpTagger(patterns)
tags = regexp_tagger.tag(tokens)
print(tags)

# Lookup tagger
print('\n\nLookup tagger\n')
fd = nltk.FreqDist(brown.words(categories='news'))
cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
most_freq_words = fd.most_common(100)
likely_tags = dict((word, cfd[word].max()) for (word, cant) in most_freq_words)
baseline_tagger = nltk.UnigramTagger(model=likely_tags,backoff=default_tagger)
tags = baseline_tagger.tag(tokens)
print(tags)

# N-Gram tagger
print('\n\nN-Gram tagger\n')
brown_tagged_sents = brown.tagged_sents(categories = 'news')
unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
tags = unigram_tagger.tag(tokens)
print(tags)

size = int(len(brown_tagged_sents) * 0.9)
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]
unigram_tagger = nltk.UnigramTagger(train_sents)
test = unigram_tagger.evaluate(test_sents)
print(test)

# Bigram Tagger
print('\n\nBigram tagger\n')
bigram_tagger = nltk.BigramTagger(train_sents)
tags = bigram_tagger.tag(tokens)
print(tags)

# Combining taggers
print('\n\nCombined Taggers\n')
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff = t0)
t2 = nltk.BigramTagger(train_sents, backoff = t1)
tags = t2.tag(tokens)
print(tags)
test = t2.evaluate(test_sents)
print(test)

# Evaluating performance
print('\n\nEvaluating Performance\n')
test_tags = [tag for sent in brown.sents(categories = 'editorial')
                for (word,tag) in t2.tag(sent)]
gold_tags = [tag for (word,tag) in brown.tagged_words(categories = 'editorial')]
print(nltk.ConfusionMatrix(gold_tags,test_tags))

# Spanish tagger

# Functions to save files
def save_structure(structure,filename):
    file = open(filename,'wb')
    dump(structure,file)
    file.close()

def load_structure(filename):
    file = open(filename,'rb')
    structure = load(file)
    file.close()
    return structure

print('\n\nSpanish Tagger\n')
raw = 'Mi nombre es Carlos Armando y yo amo mucho a mi novia Gaby y la estaré viendo pronto'
tokens = nltk.word_tokenize(raw)
cess_tagged_sents = cess_esp.tagged_sents()
size = int(len(cess_tagged_sents) * 0.9)
train_sents = cess_tagged_sents[:size]
test_sents = cess_tagged_sents[size:]

patterns = [
    (r'.*ando$','v'),
    (r'.*endo$','v'),
    (r'.*rán$','v'),
    (r'.*esen$','v'),
    (r'.*esemos','v')
]

t0 = nltk.DefaultTagger('n')
t1 = nltk.RegexpTagger(patterns,backoff = t0)
t2 = nltk.UnigramTagger(train_sents, backoff = t1)
t3 = nltk.BigramTagger(train_sents, backoff = t2)
save_structure(t3,'tagger.pkl')
t4 = load_structure('tagger.pkl')
tags = t4.tag(tokens)
print(tags)
test = t4.evaluate(test_sents)
print(test)
