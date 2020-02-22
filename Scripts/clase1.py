from bs4 import BeautifulSoup
import nltk

f = open('Corpus/e961024.htm', encoding = 'utf-8')
text = f.read()
print(type(text))
print(len(text))
print(text[:1000])
f.close()

soup = BeautifulSoup(text,'lxml')
text = soup.get_text()
print("\nText from BSoup:\n")
print(text)

print("Length: " + str(len(text)))

words = nltk.word_tokenize(text)

print('words: \n')
print(words[:200])
