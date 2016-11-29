import time
import nltk
import pickle
import my_tokenizer
from sklearn.cross_validation import KFold, cross_val_score
from hbase_manager import HBase


class TrainingModel(object):

    def __init__(self):
        self.DATA_FILE = "/Users/binli/PycharmProjects/TweetsSentiment/data/testdata.manual.2009.06.14.csv"
        self.training_data = my_tokenizer.construct_training_data(self.DATA_FILE)
        self.word_features = my_tokenizer.get_word_features(self.training_data)


    def extract_features(self, tweet):
        tweet_words = set(tweet)
        features = {}
        for word in self.word_features:
            features["contains %s" % word] = (word in tweet_words)
        return features


    def create_classifier(self, extract, data):
        training_set = nltk.classify.apply_features(extract, data)
        classifier = nltk.NaiveBayesClassifier.train(training_set)
        # print(len(training_set))
        #
        # print(classifier.show_most_informative_features(32))

        # # cross validation
        # cv = KFold(len(training_set), n_folds=10,
        #                             shuffle=True, random_state=None)
        # for traincv, testcv in cv:
        #     classifier = nltk.NaiveBayesClassifier.train(training_set[traincv[0]:traincv[len(traincv) - 1]])
        #     print 'accuracy:', nltk.classify.util.accuracy(classifier, training_set[testcv[0]:testcv[len(testcv) - 1]])

        # accuracy: 0.968220338983
        # accuracy: 0.969072164948
        # accuracy: 0.969262295082
        # accuracy: 0.969450101833
        # accuracy: 0.969387755102
        # accuracy: 0.969387755102
        # accuracy: 0.970772442589
        # accuracy: 0.96963562753
        # accuracy: 0.969387755102
        # accuracy: 0.968819599109

        return classifier


    def write_to_hbase(self, conn, batch, tweet, sentiment):
        '''
        put tweet and its sentiment into HBase.

        rowkey:: candidate:timestamp; column family: data; column1: tweet; column 2: sentiment
        dumbo: 128.122.215.51

        :param classifier:
        :param tweet: line of tweets formatted file
        :return:
        '''
        tweet_splited = tweet.split('|||')
        tweet_splited.append(sentiment)
        HBase.insert_row(batch, tweet_splited)


if __name__ == "__main__":
    tweets_file = "/Users/binli/PycharmProjects/TweetsSentiment/data/hillari_tweets.txt"
    # positive
    tweet = "you are awesome"
    # negative
    tweet_1 = "I hate you"
    # neutral
    tweet_2 = "eat"

    # classifier = create_classifier(extract_features, training_data)


    # save classifier
    # f = open('my_classifier.cf', 'wb')
    # pickle.dump(classifier, f)
    # f.close()

    # load classifier
    f = open('my_classifier.cf', 'rb')
    classifier = pickle.load(f)
    print(classifier.show_most_informative_features(32))
    f.close()

    model = TrainingModel()
    conn, batch = HBase.connect_to_hbase()
    print("Connected to HBase successfully. Table name: %s", HBase.table_name)

    try:
        with open(tweets_file) as tweets:
            for tweet in tweets:
                # TODO: parse the third field of formatted tweet
                tweet_cleaned = my_tokenizer.tokenize(tweet.split('|||')[2])
                if tweet_cleaned:
                    sentiment = classifier.classify(model.extract_features(tweet_cleaned))
                    model.write_to_hbase(conn, batch, tweet, sentiment)
                    HBase.row_count += 1
        batch.send()
    finally:
        conn.close()

    print("done. inserted %i rows to %s:%s", (HBase.row_count, HBase.namespace, HBase.table_name))



