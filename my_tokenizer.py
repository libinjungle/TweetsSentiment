import re
import string
import tokenizer
import preprocessor as p
import nltk
from nltk.corpus import stopwords


# download nltk module
nltk.download('stopwords')
nltk.download('punkt')
operators = {'not'}
# stop words
stops = set(stopwords.words('english')) - operators
porter_stemmer = nltk.stem.PorterStemmer()
# tok = tokenizer.Tokenizer(preserve_case=False)
p.set_options(p.OPT.URL, p.OPT.HASHTAG)


def tokenize(text):
    # preprocessing tweet
    pre_cleaned = p.clean(text)
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
    tweet_cleaned = map(lambda s: s.translate(None, string.punctuation), tweet_cleaned)
    # remove stop word
    tweet_cleaned_no_stopword = [word for word in tweet_cleaned if word not in stops]
    # apply porter stemmer
    tweet_stemmer = map(lambda word: str(porter_stemmer.stem_word(word)), tweet_cleaned_no_stopword)
    tweet_final = [word for word in tweet_stemmer if len(word) >= 2]
    return tweet_final


def construct_training_data(filename):
    training_data = []
    with open(filename) as tweets:
        for tweet in tweets:
            # get tweet text
            text_splited = tweet.split('"')
            tweet_text = text_splited[-2].strip('"').lower()
            tweet_final = tokenize(tweet_text)
            if text_splited[1] == '0':
                training_data.append((tweet_final, 'negative'))
            elif text_splited[1] == '2':
                training_data.append((tweet_final, 'neutral'))
            elif text_splited[1] == '4':
                training_data.append((tweet_final, 'positive'))
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
    file = "/Users/binli/PycharmProjects/TweetSentiment/data/testdata.manual.2009.06.14.csv"
    construct_training_data(file)


