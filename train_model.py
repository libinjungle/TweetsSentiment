import nltk
import pickle
import my_tokenizer
from sklearn.cross_validation import KFold, cross_val_score

DATA_FILE = "/Users/binli/PycharmProjects/TweetSentiment/data/testdata.manual.2009.06.14.csv"
training_data = my_tokenizer.construct_training_data(DATA_FILE)
word_features = my_tokenizer.get_word_features(training_data)


def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in word_features:
        features["contains %s" % word] = (word in tweet_words)
    return features


def create_classifier(extract, data):
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

if __name__ == "__main__":
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

    print(classifier.classify(extract_features(tweet_2.split())))













