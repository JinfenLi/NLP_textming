import nltk

f = open("desert.txt")
raw_text = f.read()
f.close()
tokens = nltk.word_tokenize(raw_text)
text = nltk.Text(tokens)
# print(text.concordance('ice'))
porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()
desertPstem = [porter.stem(t) for t in tokens]
print(desertPstem[:26])
desertLstem = [lancaster.stem(t) for t in tokens]
print(desertLstem[:26])

wnl = nltk.WordNetLemmatizer()
desertLemma = [wnl.lemmatize(t) for t in tokens]
print(desertLemma[:26])

import re
string = "Mr. Black and Mrs. Brown attended the lecture by Dr. Gray, but Gov. White wasn't there."
string_tokens = nltk.regexp_tokenize(string, r"\w+\. |\w+'\w|\w+")
print(string_tokens)