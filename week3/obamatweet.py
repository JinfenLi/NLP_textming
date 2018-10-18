import pandas as pd
import re
from nltk import FreqDist
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt


def readcsv(csv):

    df = pd.read_csv(csv,encoding='utf-8',sep=",")
    gp = df.groupby('date')
    return df, gp


def splitdate(date):
    datelist = date.split(" ")
    year = ''
    month = ''
    day = ''
    if len(datelist)== 2:
        month = datelist[0]
        day = datelist[1]
        year = '2016'
    elif len(datelist) == 3:
        month = datelist[1]
        day = datelist[0]
        year = datelist[2]

    return year+","+month+","+day


def removelink(text):
    return re.sub(r"http://\S+",'',text)



def freq_term_each_year(df):
    gf_year = df.groupby("year")
    for i in range(2007,2017):
        year_text = " ".join(gf_year.get_group(str(i))["text"].values)
        stop = set(stopwords.words('english'))
        other_stop = ['watch','obama','today','get','make','time','et','rt','mitt']
        stop = stop | set(other_stop)
        emmatokens = nltk.word_tokenize(year_text)
        emmatokens = [w.lower() for w in emmatokens]
        emmatokens = [w for w in emmatokens if w.isalpha()]
        emmatokens = [w for w in emmatokens if w not in stop]
        fd = FreqDist(emmatokens).most_common(10)
        print("year ",i)
        print(year_text)
        print(fd)
        print(list(map(lambda x:x[0],fd)))
        counter_list = sorted(fd, key=lambda x: x[1], reverse=True)
        # print(counter_list)
        label = list(map(lambda x: x[0], counter_list[:50]))
        value = list(map(lambda y: y[1], counter_list[:50]))
        plt.bar(range(len(value)), value, tick_label=label)
        plt.title("the frequent terms in %d"%i)
        # plt.show()



def hashtag(text):
    return " ".join(re.findall(r'#\w+',text)).replace("#","")


def get_topics_2011(df):
    gf_year = df.groupby("year")
    year_text = " ".join(gf_year.get_group('2012')["text"].values)
    topics = hashtag(year_text)
    print(topics.split())
    fd = FreqDist(topics.split())
    print(fd.most_common(10))

def main():
    df, gf = readcsv("BarackObama.csv")

    # map year,month,day
    df['date'] = df['date'].map(splitdate)
    df["year"] = df["date"].map(lambda x: x.split(",")[0])  # 分别处理新旧两列
    df["month"] = df["date"].map(lambda x: x.split(",")[1])
    df['day'] = df["date"].map(lambda x: x.split(",")[2])

    # group by year,month
    gf = df.groupby(["year", "month"]).size().reset_index()
    gf.columns = ['year', 'month', 'count']
    gf.to_csv("gf.csv")

    # clean text
    df['text'] = df['text'].map(removelink)

    freq_term_each_year(df)

    # extract topics
    df['topic'] = df['text'].map(hashtag)
    fdist = FreqDist(df['topic'].values)
    print(df['topic'].values)
    topkeys = fdist.most_common(50)[1:]
    topic_df = pd.DataFrame(topkeys)
    topic_df.to_csv("topic.csv")

    #topic focus in oct, 2011
    get_topics_2011(df)

if __name__ == '__main__':
    main()




