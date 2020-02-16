import nltk
from nltk.corpus import brown, treebank

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
