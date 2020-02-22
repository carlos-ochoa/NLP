from nltk.stem import PorterStemmer, LancasterStemmer, RegexpStemmer, SnowballStemmer

# Porter Stemmer
ps = PorterStemmer()
print(ps.stem('jumping'),ps.stem('jumps'),ps.stem('jumped'))
print(ps.stem('lying'))
print(ps.stem('strange'))

# Lancaster Stemmer
ls = LancasterStemmer()
print(ls.stem('jumping'),ls.stem('jumps'),ls.stem('jumped'))
print(ls.stem('lying'))
print(ls.stem('strange'))

# Regexp Stemmer
rs = RegexpStemmer('ing$|s$|ed$', min = 4)
print(rs.stem('jumping'),rs.stem('jumps'),rs.stem('jumped'))
print(rs.stem('lying'))
print(rs.stem('strange'))

# Snowball Stemmer
ss = SnowballStemmer('spanish')
print(ss.stem('saltar'),ss.stem('contrariedad'),ss.stem('actividad'))
