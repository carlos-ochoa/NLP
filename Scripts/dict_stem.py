# Script for reading the generate.txt file and create a lemme dictionary using pickle
from pickle import dump

dictionary = {}

file = open('generate.txt','r', encoding = 'latin-1')
# Generate a list of lines
lines = file.readlines()
# Iterate over the list to split its contents
for line in lines:
    print(line)
    info = tuple(line.split())
    print(info)
    '''
        We need 3 items:
            tuple[0] is a word, we have to find index of char '#'
            tuple[-2] is the tag
            tuple[-2][0] is the simplified tag
    '''
    if len(info) > 0:
        index = info[0].find('#')
        stem = info[0][:index]
        word = info[0].replace('#','')
        tag = info[-2][0].lower()
        key = word
        dictionary[key] = stem
# Save the dictionary
print(dictionary)
file2 = open('stem_dict.pkl','wb')
dump(dictionary,file2)
file2.close()
