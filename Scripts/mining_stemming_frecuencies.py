import WordMiner as wm

FILE_PIK_VOC = '../Files/cosines_s_p.pkl'
FILE_PIK_VECT = '../Files/vectors_s_p.pkl'
FILE_COS = '../Files/cosines_s_p.txt'

vocabulary = wm.load_structure('../Files/Vocabularies/vocabulary_s2.pkl')
contexts = wm.load_structure('../Files/Contexts/contexts_s2.pkl')
#vectors = wm.vectorize_tokens_c(contexts,vocabulary)
#vectors = wm.vectorize_frec(contexts,vocabulary)
vectors = wm.vectorize_tf_idf(contexts,vocabulary)
print('\n\nVectorization done!\n')
#vectors = wm.load_structure('vectors.pkl')
print('\n\nPlease type the word you want to take as reference in cosine calculation...\n')
word = input()
cosines = wm.calculate_cosines(word,vectors,use_tags = False)
print("\n\nCosines calculated!\n")
