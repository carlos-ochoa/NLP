import WordMiner as wm

FILE_PIK_VOC = '../Files/Vocabularies/vocabulary_l.pkl'
FILE_PIK_VECT = '../Files/Vectors/vectors_s_tf_2.pkl'
FILE_COS = '../Files/cosines_s_p.txt'

vocabulary = wm.load_structure(FILE_PIK_VOC)
sentences = wm.load_structure('../Files/Tokens/sentences.pkl')
print('\n\nPlease type the word you want to take as reference in cosine calculation...\n')
word = input()
results = wm.discover_syntagmatic_relations(word,sentences,vocabulary, threshold = 0.12)
print("\n\nCosines calculated!\n")
