import pandas as pd
from scipy.stats import itemfreq
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import nltk.stem

data = pd.read_csv("./kaggle/train.tsv", delimiter='\t')


def choose_vec(type):
    vec = ''
    if type == 'unigram_boolean':
        vec = CountVectorizer(binary=True, min_df=5, stop_words='english')
    elif type == 'unigram_count':
        vec = CountVectorizer(binary=False, min_df=5, stop_words='english')
    elif type == 'gram12_count':
        vec = CountVectorizer(ngram_range=(1,1), min_df=5, stop_words='english', binary=False)
    elif type == 'unigram_tfidf':
        vec = TfidfVectorizer(use_idf=True, min_df=5, stop_words='english')
    elif type == 'stem':
        english_stemmer = nltk.stem.SnowballStemmer('english')

        class StemmedCountVectorizer(CountVectorizer):
            def build_analyzer(self):
                analyzer = super(StemmedCountVectorizer, self).build_analyzer()
                return lambda doc: [english_stemmer.stem(w) for w in analyzer(doc)]

        vec = StemmedCountVectorizer(min_df=3, analyzer="word")

    return vec


def train(vec, model):
    # train_data = pd.read_csv("./kaggle/train.tsv")
    # test_data = pd.read_csv("./kaggle/test.tsv")
    y = data['Sentiment'].values
    X = data['Phrase'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    training_category_dist = itemfreq(y_train)
    print("distribution ", training_category_dist)
    vectorize = choose_vec(vec)
    X_train_vec = vectorize.fit_transform(X_train)
    # initialize the LinearSVC model

    model.fit(X_train_vec, y_train)

    return X_train, X_test, y_train, y_test, model, vectorize, X, y


def interpretemodel(X_train, X_test, y_train, y_test, model, vectorize):
    # print(vectorize.vocabulary_)
    # for i in range(0, 2):
    #     print(model.feature_log_prob_[i][vectorize.vocabulary_.get('amazing')])
    feature_ranks = sorted(zip(model.coef_[0], vectorize.get_feature_names()))
    very_negative_features = feature_ranks[-10:]
    # print(very_negative_features)
    log_ratios = []
    features = vectorize.get_feature_names()
    vneg_cond_prob = model.coef_[0]
    vpos_cond_prob = model.coef_[4]

    for i in range(0, len(features)):
        log_ratio = vpos_cond_prob[i] - vneg_cond_prob[i]
        log_ratios.append(log_ratio)

    exercise_C_ranks = sorted(zip(log_ratios, features))
    print("10 indicative positive words",list(map(lambda x:x[1],exercise_C_ranks))[:10])
    print("10 indicative negative words", list(map(lambda x:x[1],exercise_C_ranks))[-10:])


def test(model, X_test, y_test, vectorize):
    X_test_vec = vectorize.transform(X_test)
    y_pred = model.predict(X_test_vec)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    target_names = ['negative', 'somewhat negative','neutral','somewhat positive','positive']
    print(classification_report(y_test, y_pred, target_names=target_names))
    # output = open('linearSVC_prediction_output.csv', 'w')
    # for x, value in enumerate(y_pred):
    #     output.write(str(value) + '\n')
    # output.close()


def pipelinet(X, y):
    bNB_pipe = Pipeline([('vect', CountVectorizer(binary=True)), ('bernNB', BernoulliNB())])
    scores = cross_val_score(bNB_pipe, X, y, cv=3)
    print(sum(scores) / len(scores))

    bNB_pipe = Pipeline([('vect', CountVectorizer(binary=True, stop_words='english')), ('bernNB', BernoulliNB())])
    scores = cross_val_score(bNB_pipe, X, y, cv=3)
    print(sum(scores) / len(scores))

    bNB_pipe = Pipeline(
        [('vect', CountVectorizer(binary=True, stop_words='english', ngram_range=(1, 2))), ('bernNB', BernoulliNB())])
    scores = cross_val_score(bNB_pipe, X, y, cv=3)
    print(sum(scores) / len(scores))

    ##MNB TFIDF
    mNB_tfidf_pipe = Pipeline(
        [('nb_tf', TfidfVectorizer(use_idf=True, binary=False)), ('nb', MultinomialNB())])
    scores = cross_val_score(mNB_tfidf_pipe, X, y, cv=3)
    print(sum(scores) / len(scores))

    ##MNB TF
    mNB_tf_pipe = Pipeline(
        [('nb_tf', TfidfVectorizer(use_idf=False, binary=False, stop_words='english')), ('nb', MultinomialNB())])
    scores = cross_val_score(mNB_tf_pipe, X, y, cv=3)
    print(sum(scores) / len(scores))

    ##MNB with Bool
    nb_clf_pipe = Pipeline(
        [('vect', CountVectorizer(binary=False, stop_words='english', ngram_range=(1, 2))), ('nb', MultinomialNB())])
    scores = cross_val_score(nb_clf_pipe, X, y, cv=3)
    avg = sum(scores) / len(scores)
    print(avg)

def kaggle():# read in the test data
    english_stemmer = nltk.stem.SnowballStemmer('english')

    class StemmedCountVectorizer(CountVectorizer):
        def build_analyzer(self):
            analyzer = super(StemmedCountVectorizer, self).build_analyzer()
            return lambda doc: [english_stemmer.stem(w) for w in analyzer(doc)]

    vec = StemmedCountVectorizer(min_df=3, analyzer="word", stop_words='english')
    # vec = CountVectorizer(ngram_range=(1, 2), min_df=5, stop_words='english', binary=False)
    train_data = pd.read_csv("./kaggle/train.tsv",delimiter='\t')
    X_train = train_data['Phrase'].values
    y_train = train_data['Sentiment'].values
    X_train_vec = vec.fit_transform(X_train)
    clf = LinearSVC(C=1)

    kaggle_test=pd.read_csv("./kaggle/test.tsv", delimiter='\t')

    # preserve the id column of the test examples
    kaggle_ids=kaggle_test['PhraseId'].values

    # read in the text content of the examples
    kaggle_X_test=kaggle_test['Phrase'].values

    # vectorize the test examples using the vocabulary fitted from the 60% training data
    kaggle_X_test_vec=vec.transform(kaggle_X_test)

    # predict using the NB classifier that we built
    kaggle_pred=clf.fit(X_train_vec, y_train).predict(kaggle_X_test_vec)

    # combine the test example ids with their predictions
    kaggle_submission=zip(kaggle_ids, kaggle_pred)

    # prepare output file
    outf=open('kaggle/kaggle_submission_linearSVC.csv', 'w')

    # write header
    outf.write('PhraseId,Sentiment\n')

    # write predictions with ids to the output file
    for x, value in enumerate(kaggle_submission): outf.write(str(value[0]) + ',' + str(value[1]) + '\n')

    # close the output file
    outf.close()

def main():
    model = MultinomialNB()
    # model = BernoulliNB()
    # initialize the LinearSVC model
    model = LinearSVC(C=1)

    # use the training data to train the model

    # X_train, X_test, y_train, y_test, model, vectorize, X, y = train('gram12_count', model)
    # interpretemodel(X_train, X_test, y_train, y_test, model, vectorize)
    # test(model, X_test, y_test, vectorize)

    kaggle()



if __name__ == '__main__':
    main()
