import nltk
import my_tokenizer

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
    print(classifier.show_most_informative_features(32))
    return classifier


if __name__ == "__main__":
    tweet = "you are awesome"
    classifier = create_classifier(extract_features, training_data)
    print(classifier.classify(extract_features(tweet.split())))










