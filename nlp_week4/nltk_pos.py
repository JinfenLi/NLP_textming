import nltk
from nltk.corpus import brown
from nltk.corpus import treebank
print(brown.tagged_sents()[:2])
print(brown.tagged_words()[:50])
wordtag = brown.tagged_words()[0]
brown_humor_tagged = brown.tagged_words(categories='humor', tagset='universal')
print(brown_humor_tagged[:50])
a = nltk.corpus.nps_chat.tagged_words()[:50]
print(a)

treebank_tokens = treebank.words()
print("treebank_tokens ", treebank_tokens)
treebank_tagged_words = treebank.tagged_words()[:50]

print("tree tagged", treebank_tagged_words[:50])

treebank_tagged = treebank.tagged_sents()[:2]
print(treebank_tagged[:2])


tag_fd = nltk.FreqDist(tag for (word, tag) in treebank_tagged_words)
for tag,freq in tag_fd.most_common():
    print (tag, freq)
