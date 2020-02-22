import TextPreprocesser

FILE = '../Files/Corpus/e961024.htm'
FILE_VOC = '../Files/Vocabularies/vocabulary_s2.txt'
FILE_TOK = '../Files/Tokens/tokens_s2.txt'
FILE_PIK_TOK = '../Files/Tokens/tokens_f.pkl'

filtered_tokens = TextPreprocesser.load_structure(FILE_PIK_TOK)
print('\n\nStemmings\n')
stem_tokens = TextPreprocesser.stemming_tokens_ss(filtered_tokens)
print(stem_tokens[:400])
vocabulary = TextPreprocesser.get_vocabulary(stem_tokens)
print('\n\nThis is the vocabulary used in the text (First 200 words) : \n')
print(vocabulary)
TextPreprocesser.write_file(vocabulary,FILE_VOC)

contexts = TextPreprocesser.get_context_c(stem_tokens)
print('\n\nText Preprocessing done!\n')
