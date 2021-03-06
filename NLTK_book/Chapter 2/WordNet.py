from nltk.corpus import wordnet as wn

motorcar = wn.synset('car.n.01')
print(wn.synsets('motorcar'))
print(wn.synset('car.n.01').lemma_names)
print(wn.synset('car.n.01').definition)
print(wn.synset('car.n.01').examples)

print(motorcar.hyponyms())
print(motorcar.hypernyms())
print(motorcar.part_meronyms())
print(motorcar.substance_meronyms())
print(motorcar.member_holonyms())
print(motorcar.entailments())

print("\n\nSemantic Similarity\n")
right = wn.synset('right_whale.n.01')
orca = wn.synset('orca.n.01')
minke = wn.synset('minke_whale.n.01')
tortoise = wn.synset('tortoise.n.01')
novel = wn.synset('novel.n.01')
print(right.lowest_common_hypernyms(minke))
print(right.lowest_common_hypernyms(orca))
print(right.lowest_common_hypernyms(tortoise))
print(right.lowest_common_hypernyms(novel))

print('\nDepths of synsets found\n')
print(wn.synset('baleen_whale.n.01').min_depth())
print(wn.synset('whale.n.02').min_depth())
print(wn.synset('vertebrate.n.01').min_depth())
print(wn.synset('entity.n.01').min_depth())

print("\nPath similarities\n")
print(str(right.path_similarity(minke)))
print(str(right.path_similarity(orca)))
print(str(right.path_similarity(tortoise)))
print(str(right.path_similarity(novel)))
