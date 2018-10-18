import pandas as pd
import math
import re


def top_indicative_words(csv, type):

    senti_output = pd.read_csv(csv, delimiter='\t')
    features = senti_output['token'].values
    print(senti_output.columns)
    pos_cond_prob = []
    neg_cond_prob = []
    if type == 'senti':
        neg_cond_prob = senti_output['negative'].values
        pos_cond_prob = senti_output['positive'].values
    elif type == 'lie':
        neg_cond_prob = senti_output['fake'].values
        pos_cond_prob = senti_output['true'].values

    log_ratios = list(map(lambda x,y:math.log(x)-math.log(y),pos_cond_prob,neg_cond_prob))


    features_ranks = sorted(zip(log_ratios, features))

    features_ranks = list(filter(lambda x:re.match(r'[a-zA-Z]+',x[1]),features_ranks))
    top_pos_features = features_ranks[-20:]
    top_neg_features = features_ranks[:21]
    if type == "senti":
        print("top positive words ",list(map(lambda x:x[1],top_pos_features)))
        print("top negative words ", list(map(lambda x:x[1],top_neg_features)))
    elif type == "lie":
        print("top true words ", list(map(lambda x:x[1],top_pos_features)))
        print("top fake words ", list(map(lambda x:x[1],top_neg_features)))
    return top_pos_features, top_neg_features

def main():

    top_indicative_words("sentiment_ignorecase.txt", "senti")

    top_indicative_words("lie_ignorecase.txt", "lie")




if __name__ == '__main__':
    main()