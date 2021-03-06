import TextPreprocesser

FILE = 'Corpus/e961024.htm'
FILE_VOC = 'vocabulary_l.txt'
FILE_TOK = 'tokens_l.txt'

text = TextPreprocesser.load_html_file(FILE)
print('Archivo cargado : ' + FILE)
cleaned_text = TextPreprocesser.clean_html_text(text)
print('\nText without html (First 300 characters) : \n' + cleaned_text[:300])
tokens = TextPreprocesser.tokenize_text(cleaned_text)
print('\nTokenization of this text (First 200 tokens) : \n')
print(tokens[:200])
cleaned_tokens = TextPreprocesser.remove_characters_after_tokenization(tokens)
print('\nSpecial characters removed : \n')
print(cleaned_tokens[:200])
print('\n\nLength of tokens list: ' + str(len(cleaned_tokens)))
filtered_tokens = TextPreprocesser.remove_stopwords(cleaned_tokens)
print('\nStopwords removed : \n')
print(filtered_tokens[:200])
print('\n\nLength of new tokens list: ' + str(len(filtered_tokens)))
print('\n\nTagged tokens\n')
tagged_tokens = TextPreprocesser.tag_tokens(filtered_tokens)
print(tagged_tokens[:200])
TextPreprocesser.write_file(tagged_tokens,FILE_TOK)
lem_tokens = TextPreprocesser.lematize_tokens(tagged_tokens)
print('\n\nLematized tokens\n')
print(lem_tokens[:200])
print('\nLength of lem_tokens: ' + str(len(lem_tokens)))
vocabulary = TextPreprocesser.get_vocabulary(lem_tokens)
print('\n\nThis is the vocabulary used in the text (First 200 words) : \n')
print(vocabulary)
TextPreprocesser.write_file(vocabulary,FILE_VOC)
print('\n\nLength of vocabulary : ' + str(len(vocabulary)))
#contexts = TextPreprocesser.get_context(filtered_tokens)
#print('\n\nContexts:\n')
#print(contexts['bajar'])
contexts = TextPreprocesser.get_context_c(lem_tokens)
print('\n\nText Preprocessing done!\n')
