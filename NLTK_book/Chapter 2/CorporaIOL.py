import nltk
from nltk.corpus import udhr,gutenberg

languages = ['Chickasaw','English','Spanish','German_Deutsch','Greenlandic_Inuktikut','Hungarian_Magyar','Ibibio_Efik']
cfd = nltk.ConditionalFreqDist(
    (lang,len(word))
    for lang in languages
    for word in udhr.words(lang + '-Latin1')
)

cfd.plot(cumulative = True)

print("\n\nAccesing in different ways to corpus\n")

raw = gutenberg.raw("burgess-busterbrown.txt")
words = gutenberg.words("burgess-busterbrown.txt")
sents = gutenberg.sents("burgess-busterbrown.txt")
print(raw[:20])
print(words[:20])
print(sents[:20])
