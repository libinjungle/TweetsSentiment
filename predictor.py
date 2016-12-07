import numpy
from my_classifiers import *
from corpus_summary import *


TWEETS_FILE = "/Users/binli/PycharmProjects/TweetsSentiment/data/tweets_cleaned.txt"
TRAIN_DATA_FILE = "/Users/binli/PycharmProjects/TweetsSentiment/data/training.1600000.processed.noemoticon.csv"
TRUMP_BASELINE_PREDICTION = "/Users/binli/PycharmProjects/TweetsSentiment/data/stats/prediction/trump_baseline_prediction"
TRUMP_TWEETS_FILTERED_CLEANED = "/Users/binli/PycharmProjects/TweetsSentiment/data/trump_tweets_filtered_large.txt"

HILLARY_BASELINE_PREDICTION = "/Users/binli/PycharmProjects/TweetsSentiment/data/stats/prediction/hillary_baseline_prediction"
HILLARY_TWEETS_FILTERED_CLEANED = "/Users/binli/PycharmProjects/TweetsSentiment/data/hillary_tweets_filtered_large.txt"


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
    '''
    vectorize each tweet
    :param cf:
    :param vectorizer:
    :param TWEETS_FILE: both filtered and cleaned tweets file used for prediction.
                        we need to do filtration because after preprocessing, some
                        tweets may be reduced to zero length. only non-zero length
                        tweets are kept to do the prediction
    :return:
    '''
    results = []
    with open(TWEETS_FILE) as f:
        for i, line in enumerate(f):
            tweet = line.split()
            # print(tweet)
            # False means tweet line does not label
            v = vectorizer.vectorize(tweet, False)
            label = cf.classify(v)
            # print(i, label)
            results.append(label)
    arr = numpy.array(results)
    ans = numpy.reshape(arr, (len(results), 1))
    print('my sentiment:')
    print(ans)
    return ans


def baseline_sentiment(baseline_file):
    results = []
    with open(baseline_file) as baseline:
        for line in baseline:
            results.append(lb_to_int[line.rstrip()])
    arr = numpy.array(results)
    ans = numpy.reshape(arr, (len(results), 1))
    print('baseline: ')
    print(ans)
    return ans


def compare_to_baseline(baseline, myresults):
    tp = Counter()
    tn = Counter()
    fp = Counter()
    fn = Counter()

    for i in LABELS:
        tp[i] += ((baseline == i) * (myresults == i)).sum()
        tn[i] += ((baseline != i) * (myresults != i)).sum()
        fp[i] += ((baseline != i) * (myresults == i)).sum()
        fn[i] += ((baseline == i) * (myresults != i)).sum()
    return tp, tn, fp, fn


if __name__ == '__main__':
    cf, vectorizer = gen_classifier()

    # for trump tweets
    # my_trump_senti = my_sentiment(cf, vectorizer, TRUMP_TWEETS_FILTERED_CLEANED)
    # trump_baseline_senti = baseline_sentiment(TRUMP_BASELINE_PREDICTION)

    # for hillary tweets
    my_hillary_senti = my_sentiment(cf, vectorizer, HILLARY_TWEETS_FILTERED_CLEANED)
    hillary_baseline_senti = baseline_sentiment(HILLARY_BASELINE_PREDICTION)
    tp, tn, fp, fn = compare_to_baseline(hillary_baseline_senti, my_hillary_senti)


    results = get_validation_summary(tp, tn, fp, fn, True)
    header = 'Label,Model(u),Freq,accuracy,precison,recall'

    ans = []
    for lb in results:
        r = results[lb]
        ans.append([lb, 0, 2, r[0], r[1], r[2]])
    numpy.savetxt('hillary_large_prediction_base_comp.csv', ans, fmt='%d,%d,%d,'+'%.10f,'*3, delimiter=',', header=header)



