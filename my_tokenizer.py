
import re
import string
import codecs
import tokenizer
import preprocessor as p
import nltk
from nltk.corpus import stopwords
from unidecode import unidecode
import time


# download nltk module. only do this on the first time
nltk.download('stopwords')
nltk.download('punkt')
not_included_operators = {'not'}
included_operators = {'rt', 'http'}
# stop words
stops = set(stopwords.words('english')).union(included_operators) - not_included_operators
porter_stemmer = nltk.stem.PorterStemmer()
# tok = tokenizer.Tokenizer(preserve_case=False)
p.set_options(p.OPT.URL, p.OPT.HASHTAG)


def tokenize(text):
    # preprocessing tweet
    pre_cleaned = p.clean(text.lower())
    # remove quotes
    temp = re.sub(r'&amp;quot;|&amp;amp', '', pre_cleaned)
    # remove citations
    temp = re.sub(r'@[a-zA-Z0-9]*', '', temp)
    # remove tickers
    temp = re.sub(r'\$[a-zA-Z0-9]*', '', temp)
    # remove numbers
    temp = re.sub(r'[0-9]*', '', temp)

    tweet_list = nltk.word_tokenize(temp)
    # remove punctuation word
    tweet_cleaned = filter(lambda x: x not in string.punctuation, tweet_list)
    # remove punctuation inside word
    # tweet_cleaned = map(lambda s: s.translate(None, string.punctuation), tweet_cleaned)
    tweet_cleaned = map(lambda s: s.translate({ord(c) : None for c in string.punctuation}), tweet_cleaned)
    # remove stop word
    tweet_cleaned_no_stopword = [word for word in tweet_cleaned if word not in stops]
    # apply porter stemmer. note: http is porter stemmer of some other word
    tweet_stemmer = map(lambda word: porter_stemmer.stem_word(unidecode(word)), tweet_cleaned_no_stopword)

    tweet_final = [word.lower() for word in tweet_stemmer if len(word) >= 2]
    return tweet_final


def clean_tweet(ifile, ofile):
    '''
    clean tweet and output to a new file.

    :param ifile:
    :param ofile:
    :return:
    '''

    with open(ofile, 'w') as out:
        with open(ifile) as tweets:
            for tweet in tweets:
                tweet_cleaned = tokenize(tweet)
                if tweet_cleaned:
                    text = ' '.join(tweet_cleaned)
                    out.write(text + '\n')


def clean_tweet_idx(ifile, ofile):
    # given a tweet file that only contains tweet text, output to a cleaned and filtered tweet file.
    # return the idxes of tweets kept in the original file.
    # used for baseline, indexes are the line numbers of tweets that are still there after tokenization
    # indexes are uniquely applied for truncating original tweets in baseline. so that baseline and
    # my classifier use the same tweets data. Only in this way, the prediction results are comparable
    indexes = []
    with open(ofile, 'w') as out:
        with codecs.open(ifile, encoding='utf-8', errors='ignore') as tweets:
            for i, tweet in enumerate(tweets):
                tweet_cleaned = tokenize(tweet)
                if tweet_cleaned:
                    text = ' '.join(tweet_cleaned)
                    out.write(text + '\n')
                    indexes.append(i)
    # 618
    print("Total number of tweets after filtration: " + str(len(indexes)))
    return indexes


def construct_training_data(filename):
    '''
    used for parsing downloaded training data
    :param filename:
    :return:
    '''
    training_data = []
    with codecs.open(filename, encoding='utf-8', errors='ignore') as tweets:
        for i, tweet in enumerate(tweets):
            if i != 0 and i%1000 == 0:
                print('processed %d tweets' % i)
            # get tweet text
            text_splited = tweet.split('"')
            if (len(text_splited) < 2):
                continue
            tweet_text = text_splited[-2].strip('"').lower()

            # print(tweet_text)

            tweet_final = tokenize(tweet_text)
            if text_splited[1] == '0':
                training_data.append((tweet_final, -1))
            elif text_splited[1] == '2':
                training_data.append((tweet_final, 0))
            elif text_splited[1] == '4':
                training_data.append((tweet_final, 1))
    # print(training_data)
    return training_data


def get_word_features(training_data):
    word_list = []
    for (words, sentiment) in training_data:
        word_list.extend(words)
    words_freq = nltk.FreqDist(word_list)
    word_features = words_freq.keys()
    return word_features



if __name__ == "__main__":

    training_data_file = "/Users/binli/PycharmProjects/TweetsSentiment/data/testdata.manual.2009.06.14.csv"
    tweets_corpus_file = "/Users/binli/PycharmProjects/TweetsSentiment/data/tweets_small_corpus"

    hillari_file = "/Users/binli/PycharmProjects/TweetsSentiment/data/hillary_plain_tweets_large.txt"
    trump_file = "/Users/binli/PycharmProjects/TweetsSentiment/data/trump_plain_tweets_large.txt"

    # construct_training_data(hillari_file)

    # tweet preprocessing
    hillari_indexes = clean_tweet_idx(hillari_file, "/Users/binli/PycharmProjects/TweetsSentiment/data/hillary_tweets_filtered_large.txt")
    trump_indexes = clean_tweet_idx(trump_file, "/Users/binli/PycharmProjects/TweetsSentiment/data/trump_tweets_filtered_large.txt")
    # generate indexes file
    trump_idx_file = "/Users/binli/PycharmProjects/TweetsSentiment/data/trump_large_tweets_idx_kept.csv"
    hillari_idx_file = "/Users/binli/PycharmProjects/TweetsSentiment/data/hillary_large_tweets_idx_kept.csv"

    time.sleep(5)
    with open (trump_idx_file, 'w') as t_idx_f:
        for i, idx in enumerate(trump_indexes):
            if  (i+1) % 100 == 0:
                t_idx_f.write(str(idx) + '\n')
            else:
                t_idx_f.write(str(idx) + ',')
    with open(hillari_idx_file, 'w') as h_idx_f:
        for i, idx in enumerate(hillari_indexes):
            if (i+1) % 100 == 0:
                h_idx_f.write(str(idx) + '\n')
            else:
                h_idx_f.write(str(idx) + ',')




