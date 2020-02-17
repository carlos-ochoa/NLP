import WordMiner as wm

FILE_PIK_VOC = 'cosines_l.pkl'
FILE_PIK_VECT = 'vectors_l.pkl'
FILE_COS = 'cosines_l.txt'

vocabulary = wm.load_structure('vocabulary_l.pkl')
contexts = wm.load_structure('contexts_l.pkl')
vectors = wm.vectorize_tokens_c(contexts,vocabulary)
print('\n\nVectorization done!\n')
#vectors = wm.load_structure('vectors.pkl')
print('\n\nPlease type the word you want to take as reference in cosine calculation...\n')
word = input()
cosines = wm.calculate_cosines(word,vectors,use_tags = True)
print("\n\nCosines calculated!\n")
