import nltk
from nltk.tokenize import TweetTokenizer
from nltk import FreqDist
from nltk.corpus import PlaintextCorpusReader

tweettknzr = TweetTokenizer()


file0 = nltk.corpus.gutenberg.fileids()[0]
emmatext = nltk.corpus.gutenberg.raw(file0)
emmatokens = nltk.word_tokenize(emmatext)
#emmatokens2 = tweettknzr.tokenize(emmatext)
emmawords = [w.lower() for w in emmatokens]
emmavocab = sorted(set(emmawords))
revisemmawords = [w for w in emmawords if w.isalpha()]
fdist = FreqDist(revisemmawords)
topkeys = fdist.most_common(50)
#total number of samples
print(fdist.N())
print(topkeys)
print(fdist.tabulate())
fdist.plot(cumulative=True)
# for pair in topkeys:
#     print(pair)
mycorpus = PlaintextCorpusReader('.','.*\.txt')
print(mycorpus.fileids())
print(mycorpus.words())
labweekstring = mycorpus.raw('labweek.txt')
# print(mycorpus)
# print(labweekstring)