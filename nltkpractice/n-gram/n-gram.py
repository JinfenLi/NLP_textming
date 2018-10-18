import nltk
from nltk import FreqDist
import re

print(nltk.corpus.gutenberg.fileids())
file0 = nltk.corpus.gutenberg.fileids()[2]
sensetexts = nltk.corpus.gutenberg.raw(file0)
# print(sensetexts[:1000])

sensetokens = nltk.word_tokenize(sensetexts)
sensewords = [w.lower() for w in sensetokens]

sensewords2 = nltk.corpus.gutenberg.words(file0)
sensewordslowercase = [w.lower() for w in sensewords2]

print(sensewords[:50])
print(sensewordslowercase[:50])

ndist = FreqDist(sensewords)
nitems = ndist.most_common(30)

sensetextText = nltk.Text(sensewords)
# sensetextText.dispersion_plot(['sense'])


def alpha_filter(w):
    pattern = re.compile('^[^a-z]+$')
    if pattern.match(w):
        return True
    else:
        return False

alphasensewords = [w for w in sensewords if not alpha_filter(w)]
alphasensewords1 = [w for w in sensewords if w.isalpha()]
difference = set(alphasensewords)-set(alphasensewords1)
print(difference)
print("length of alphasensewords: ", len(alphasensewords))
print("length of alphasensewords1: ", len(alphasensewords1))

stopwords = nltk.corpus.stopwords.words('english')
stoppedsensewords = [w for w in sensewords if w not in stopwords]

fstop = open('Smart.English.stop','r')
stoptext = fstop.read()
fstop.close()
stopwords = nltk.word_tokenize(stoptext)

# bigram frequency distribution
# print(nltk.bigrams(sensewords))
sensebigrams = list(nltk.bigrams(sensewords))
print(sensebigrams[:20])

from nltk.collocations import *

bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(sensewords)
scored = finder.score_ngrams(bigram_measures.raw_freq)
for b in scored[:20]:
    print(b)
print('****************************')

finder.apply_word_filter(alpha_filter)
scored2 = finder.score_ngrams(bigram_measures.raw_freq)
for b in scored2[:20]:
    print(b)
print('****************************')

finder.apply_word_filter(lambda x: x in stopwords)
scored3 = finder.score_ngrams(bigram_measures.raw_freq)
for b in scored3[:20]:
    print(b)
print('****************************')

finder2 = BigramCollocationFinder.from_words(sensewords)
finder2.apply_freq_filter(2)
scored4 = finder2.score_ngrams(bigram_measures.raw_freq)
for b in scored4[:20]:
    print(b)
print('score4****************************')

finder2.apply_ngram_filter(lambda w1, w2: len(w1)<2)
scored5 = finder2.score_ngrams(bigram_measures.raw_freq)
for b in scored5[:20]:
    print(b)
print('****************************')

finder3 = BigramCollocationFinder.from_words(sensewords)
scored6 = finder3.score_ngrams(bigram_measures.pmi)
for b in scored6[:20]:
    print(b)
print('****************************')

finder3.apply_freq_filter(5)
finder3.apply_word_filter(lambda w: w in stopwords)
finder3.apply_word_filter(alpha_filter)
scored7 = finder3.score_ngrams(bigram_measures.pmi)
for b in scored7[:20]:
    print(b)
print('****************************')