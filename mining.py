import WordMiner as wm

vocabulary = wm.load_structure('vocabulary.pkl')
contexts = wm.load_structure('contexts.pkl')
vectors = wm.vectorize_tokens_c(contexts,vocabulary)
print('\n\nVectorization done!\n')
#vectors = wm.load_structure('vectors.pkl')
print('\n\nPlease type the word you want to take as reference in cosine calculation...\n')
word = input()
cosines = wm.calculate_cosines(word,vectors)
print("\n\nCosines calculated!\n")
