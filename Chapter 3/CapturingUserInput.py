import nltk

s = input("Enter some text...")
print("You typed ", len(nltk.word_tokenize(s)), "words.")
