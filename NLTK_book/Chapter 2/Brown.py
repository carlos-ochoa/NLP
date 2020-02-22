import nltk
from nltk.corpus import brown

print(brown.categories())

print("\n\nAccesing News")
news = brown.words(categories="news")
fdist = nltk.FreqDist([w.lower() for w in news])
modals = ["can","could","may","might","must","will"]
for m in modals:
    print(m + ":" + str(fdist[m]))

print("\n\nMaking a Conditional Frequency Distribution\n")
cfd = nltk.ConditionalFreqDist(
    (genre,word)
    for genre in brown.categories()
    for word in brown.words(categories=genre)
)
genres = ['news','religion','hobbies','science_fiction','romance','humor']
modals = ["can","could","may","might","must","will"]
cfd.tabulate(conditions = genres, samples = modals)
