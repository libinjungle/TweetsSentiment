import numpy
from my_classifiers import *
from corpus_summary import *


TWEETS_FILE = "/Users/binli/PycharmProjects/TweetsSentiment/data/tweets_cleaned.txt"
TRAIN_DATA_FILE = "/Users/binli/PycharmProjects/TweetsSentiment/data/testdata.manual.2009.06.14.csv"
LABELS = {-1 : 'negative', 0 : 'neutral', 1 : 'positive'}


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

def classify_many(cf, vectorizer, TWEETS_FILE):
    with open(TWEETS_FILE) as f:
        for line in f:
            tweet = line.split()
            # print(tweet)
            # False means tweet line does not label
            v = vectorizer.vectorize(tweet, False)
            print(cf.classify(v))

if __name__ == '__main__':
    cf, vectorizer = gen_classifier()
    # cf.classify_tweets(td)
    classify_many(cf, vectorizer, TWEETS_FILE)
