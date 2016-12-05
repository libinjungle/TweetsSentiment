import numpy
from my_classifiers import *
from corpus_summary import *


TWEETS_FILE = "/Users/binli/PycharmProjects/TweetsSentiment/data/tweets_cleaned.txt"
TRAIN_DATA_FILE = "/Users/binli/PycharmProjects/TweetsSentiment/data/testdata.manual.2009.06.14.csv"
BASELINE_PREDICTION = "/Users/binli/PycharmProjects/TweetsSentiment/data/baseline_prediction"
TRUMP_TWEETS_FILTERED_CLEANED = "/Users/binli/PycharmProjects/TweetsSentiment/data/trump_tweets_filtered.txt"

LABELS = {-1 : 'negative', 0 : 'neutral', 1 : 'positive'}
lb_to_int = {'negative' : -1, 'very negative' : -1, 'neutral' : 0, 'positive' : 1, 'very positive' : 1}


def gen_classifier():
    # tweet: (tweet_list, label)
    tweets = construct_training_data(TRAIN_DATA_FILE)
    tf, t_doc_f = count_tokens(tweets)
    # 2 is feature frequency
    token_map = create_gram_map(tf, 2)
    # print(token_map)
    v1 = UnigramVectorizer(token_map)
    dataset = generate_dataset_vectors(tweets, v1)
    # print(dataset)
    nb_cf = NaiveBayesClassifier(v1.tokens_size, LABELS)
    tl, td = slicing_labels_features(dataset)
    nb_cf.train(td, tl)
    return nb_cf, v1


def my_sentiment(cf, vectorizer, TWEETS_FILE):
    results = []
    with open(TWEETS_FILE) as f:
        for i, line in enumerate(f):
            tweet = line.split()
            # print(tweet)
            # False means tweet line does not label
            v = vectorizer.vectorize(tweet, False)
            label = cf.classify(v)
            # print(i, label)
            results.append(lb_to_int[label])
    arr = numpy.array(results)
    arr.reshape([len(results), 1])
    return results


def baseline_sentiment(baseline_file):
    results = []
    with open(baseline_file) as baseline:
        for line in baseline:
            results.append(lb_to_int[line.rstrip()])
    arr = numpy.array(results)
    arr.reshape([len(results), 1])
    return arr


def compare_to_baseline(baseline, myresults):
    tp = Counter()
    tn = Counter()
    fp = Counter()
    fn = Counter()

    for i in LABELS:
        tp[i] = ((baseline == i) * (myresults == i)).sum()
        tn[i] = ((baseline != i) * (myresults != i)).sum()
        fp[i] = ((baseline != i) * (myresults == i)).sum()
        fn[i] = ((baseline == i) * (myresults != i)).sum()
    return tp, tn, tp, fn


if __name__ == '__main__':
    cf, vectorizer = gen_classifier()
    my_senti = my_sentiment(cf, vectorizer, TRUMP_TWEETS_FILTERED_CLEANED)
    baseline_senti = baseline_sentiment(BASELINE_PREDICTION)
    tp, tn, fp, fn = compare_to_baseline(baseline_senti, my_senti)
    get_validation_summary(tp, tn, fp, fn, True)

