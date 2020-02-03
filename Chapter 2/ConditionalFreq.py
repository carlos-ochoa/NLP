import nltk

sent = ['In','the','beginning','God','created','the','heaven']
bigrams = nltk.bigrams(sent)
for b in bigrams:
    print(b)

print("\n\nGenerating random text")

def generate_model(cfdist,word,num=15):
    for i in range(num):
        print(word,end=' ')
        word = cfdist[word].max()

text = nltk.corpus.genesis.words('english-kjv.txt')
bigrams = nltk.bigrams(text)
cfd = nltk.ConditionalFreqDist(bigrams)

print(str(cfd['God']))
generate_model(cfd,'God')
