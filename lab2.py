import nltk
from nltk.book import *
from nltk.corpus import brown,wordnet
# Generar las gráficas de distribución de frecuencia de 1000 palabras del vocabulario de text5
fd = nltk.FreqDist(text5)
fd.plot(1000)
fd.plot(1000,cumulative=True)
# Imprimir palabras de más de longitud 30 de text5
words = [w for w in text5 if len(w) > 30]
print(words)
# Generar la distribución condicional de Brown Corpus e impŕimir las frecuencias de palabras
genres = ['news','romance','humor']
wo = ['love','hate','speak','control','feel','great','president']

fdc = nltk.ConditionalFreqDist(
    (genre,word)
    for genre in brown.categories() if genre in genres
    for word in brown.words(categories=genre) if word in wo
)

fdc.tabulate()
# Igual pudo ser fdc.tabulate(conditions=genres,samples=wo) de no haber puesto los ifs en la ConditionalFreqDist
#Imprimir los synsets de computer, machine, car, sandwich y las similitudes
print("\n\nSynsets\n")
computer = wordnet.synset('computer.n.01')
machine = wordnet.synset('machine.n.01')
car = wordnet.synset('car.n.01')
sandwich = wordnet.synset('sandwich.n.01')
l = [computer,machine,car,sandwich]
print(l)
print('\n\nSimilarities\n')
print("Computer - Machine  " + str(computer.path_similarity(machine)))
print("Car - Machine  " + str(car.path_similarity(machine)))
print("Car - Computer  " + str(car.path_similarity(computer)))
print("Computer - sandwich  " + str(computer.path_similarity(sandwich)))
