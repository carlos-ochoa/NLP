from nltk.corpus import reuters

print(reuters.fileids())
print(reuters.categories())
print(reuters.words('training/9865')[:20])
