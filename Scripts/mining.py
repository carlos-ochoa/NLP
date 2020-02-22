import WordMiner as wm

FILE_PIK_VOC = 'cosines_s.pkl'
FILE_PIK_VECT = 'vectors_s.pkl'
FILE_COS = 'cosines_s.txt'

vocabulary = wm.load_structure('vocabulary_s.pkl')
contexts = wm.load_structure('contexts_s.pkl')
vectors = wm.vectorize_tokens_c(contexts,vocabulary)
print('\n\nVectorization done!\n')
#vectors = wm.load_structure('vectors.pkl')
print('\n\nPlease type the word you want to take as reference in cosine calculation...\n')
word = input()
cosines = wm.calculate_cosines(word,vectors,use_tags = False)
print("\n\nCosines calculated!\n")
