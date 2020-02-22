from nltk.book import *

word = "monstrous"
word2 = "very"

print("The selected text is " + str(text1) + "\n\n")

print("Here is the concordance of the word : " + word + "\n\n")
text1.concordance(word)

print("\n\n Similar words to monstrous\n\n")
text1.similar(word)

words = [word,word2]
print("\n\n Common contexts for the words: " + str(words) + " with text: " + str(text2))
text2.common_contexts(words)

print("\nNow trying a dispersion plot with text : " + str(text4) + "\n")
wordsDispersion = ["freedom", "security","weapons","petrol","Mexico"]
text4.dispersion_plot(wordsDispersion)

print("\n\nFinally we will generate some text in the style of :" + str(text4))
randomText = text4.generate(wordsDispersion)
