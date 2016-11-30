from my_tokenizer import *
from collections import Counter
import numpy
import random
import math
from my_classifiers import *
from my_tokenizer import *
from vectorizer import UnigramVectorizer, BigramVectorizer

PAD = "PAD"
LABELS = {-1 : 'negative', 0 : 'neutral', 1 : 'positive'}
DATA_FILE = "/Users/binli/PycharmProjects/TweetsSentiment/data/testdata.manual.2009.06.14.csv"

def token_filter(function, tokens_map):
    return {k : tokens_map[k] for k in filter(function, tokens_map)}

def token_filter_doc(function, tokens_map, token_doc_map):
    return {k : tokens_map[k] for k in filter(function, token_doc_map)}

def token_thresh_filter(tokens_map, threshold):
    return token_filter(lambda k : tokens_map[k] >= threshold, tokens_map)

def token_and_doc_thresh_filter(tokens_map, tokens_doc_map, tok_thresh, doc_thresh):
    tokens = token_filter(lambda k : tokens_map[k] >= tok_thresh, tokens_map)
    return token_filter_doc(lambda k : tokens_doc_map[k] >= doc_thresh, tokens, tokens_doc_map)


def count_tokens(tweets):
    '''

    :param tweets: each element is ([f1, f2, f3, ...], label)
    :return:
    '''
    tf = Counter()
    t_doc_f = Counter()

    for tweet in tweets:
        tokens = tweet[0]
        token_set = set()
        for token in tokens:
            tf[token] += 1
            if token not in token_set:
                t_doc_f[token] += 1
                token_set.add(token)
    return tf, t_doc_f


def count_bigrams(tweets, k=2, thresh=3):
    '''
    get bigrams that meet threshold
    :param tweets:
    :param k:      bigram
    :param thresh:
    :return:
    '''
    if k != 2:
        k = 2
    tf, t_doc_f = count_tokens(tweets)
    tf_thresh = token_thresh_filter(tf, thresh)
    t_doc_f_thresh = token_thresh_filter(t_doc_f, thresh)
    tf_thresh = {tuple([key]) : tf_thresh[key] for key in tf_thresh}
    t_doc_f_thresh = {tuple([key]) : t_doc_f_thresh[key] for key in t_doc_f_thresh}

    tf_bimers = Counter()
    t_doc_f_bimers = Counter()

    for tweet in tweets:
        tokens = tweet[0]
        padded_tweet = [PAD for i in xrange(k-1)] + tokens + [PAD for i in xrange(k-1)]
        bimers_set = set()
        for i in xrange(k-2, len(tokens)+k-1):
            left_gramer = tuple(padded_tweet[i:i+k-1])
            gramer = tuple(padded_tweet[i:i+k])
            right_gramer = tuple(padded_tweet[i+1:i+k])
            # make sure each gram meets the threshold
            if left_gramer in tf_thresh and right_gramer in tf_thresh:
                tf_bimers[gramer] += 1.0

            if left_gramer in tf_thresh and right_gramer in tf_thresh:
                if gramer not in bimers_set:
                    t_doc_f_bimers[gramer] += 1.0
                    bimers_set.add(gramer)
    # filter bimers that meet threshold
    return token_thresh_filter(tf_bimers, thresh), token_thresh_filter(t_doc_f_bimers, thresh)


def create_gram_map(tf, thresh):
    '''
    create mapping of token to its index, each token should meet the threshold
    :param tf: token_freq
    :param thresh:
    :return:
    '''
    token_map = {}
    tf_thresh = token_thresh_filter(tf, thresh)
    for i, tok in enumerate(tf_thresh):
        token_map[tok] = i
    return token_map


def generate_dataset_vectors(tweets, vectorizer):
    data = map(lambda tweet : (tweet[1], vectorizer.vectorize(tweet)), tweets)
    rows = len(data)
    cols = len(data[0][1]) + 1
    random.shuffle(data)
    data_array = numpy.zeros([rows, cols])
    for i, t in enumerate(data):
        data_array[i, 0] = t[0]
        data_array[i, 1:] = t[1]
    return data_array


def divide(dataset, i, k=4):
    num_samples = dataset.shape[0]
    fold_size = int(math.ceil(num_samples / float(k)))
    training_range = range(0, i*fold_size) + range((i+1)*fold_size, num_samples)
    validation_range = range(i*fold_size, min((i+1)*fold_size, num_samples))
    return dataset[training_range, :], dataset[validation_range, :]


def slicing_labels_features(dataset):
    data = dataset[:, 1:]
    label = dataset[:, 0]
    return label, data


def k_fold_validation(dataset, classifier, k=10):
    tp = Counter()
    tn = Counter()
    fp = Counter()
    fn = Counter()

    correct = 0
    for i in range(k):
        training_set, validation_set = divide(dataset, i, k)
        tl, td = slicing_labels_features(training_set)
        vl, vd = slicing_labels_features(validation_set)
        classifier.train(td, tl)
        predictions = classifier.classify_tweets(vd)

        for i in LABELS:
            tp_i = ((vl == i) * (predictions == i)).sum()
            tp[i] += tp_i
            correct += tp_i
            tn_i = ((vl != i) * (predictions != i)).sum()
            tn[i] += tn_i
            correct += tn_i
            fp[i] = ((vl != i) * (predictions == i)).sum()
            fn[i] = ((vl == i) * (predictions != i)).sum()
    size = dataset.shape[0]
    # the first return element is the accuracy for all labels
    return correct / float(size), tp, tn, fp, fn


def get_validation_summary(tp, tn, fp, fn, verbose=True):
    ans = {}
    for i in tp:
        ans[i] = get_label_summary(tp, tn, fp, fn, i, verbose)
    return ans


def get_label_summary(tp, tn, fp, fn, i, verbose=True):
    accu = (tp[i]+tn[i]) / float(tp[i]+tn[i]+fp[i]+fn[i])
    precision = tp[i] / float(tp[i]+fp[i])
    recall = tp[i] / float(tp[i]+fn[i])
    if verbose:
        print('------------label:%d------------' % i)
        print('tp: %d, tn: %d, fp: %d, fn: %d' % (tp[i], tn[i], fp[i], fn[i]))
        print('accuracy:    %.3f' % accu)
        print('precison:    %.3f' % precision)
        print('recall:      %.3f' % recall)
        print('--------------------------------')
    return accu, precision, recall


if __name__ == '__main__':
    tweets = construct_training_data(DATA_FILE)
    tf, t_doc_f = count_tokens(tweets)
    bi_f, bi_doc_f = count_bigrams(tweets, 2, 2)

    ans = {}
    for i in range(2, 4):
        token_map = create_gram_map(tf, i)
        bigram_map = create_gram_map(bi_f, i)

        uni_v = UnigramVectorizer(token_map)
        bigram_v = BigramVectorizer(bigram_map)
        vectorizers = [uni_v, bigram_v]
        v_map = {0 : 'unigram', 1 : 'bigram'}
        print('************Feature Freq=%d********' % i)

        for j, v in enumerate(vectorizers):
            print(v_map[j] + ' ' + 'stats')
            dataset = generate_dataset_vectors(tweets, v)
            classifier = NaiveBayesClassifier(v.tokens_size, LABELS)
            accu, tp, tn, fp, fn = k_fold_validation(dataset, classifier)
            ans[(j, i)] = get_validation_summary(tp, tn, fp, fn)


# 10-fold validation on unigram (Naive Bayes)
# ************Feature Freq=2********
# ------------label:0------------
# tp: 89, tn: 298, fp: 9, fn: 6
# accuracy:    0.963
# precison:    0.908
# recall:      0.937
# --------------------------------
# ------------label:1------------
# tp: 129, tn: 262, fp: 6, fn: 6
# accuracy:    0.970
# precison:    0.956
# recall:      0.956
# --------------------------------
# ------------label:-1------------
# tp: 125, tn: 281, fp: 3, fn: 6
# accuracy:    0.978
# precison:    0.977
# recall:      0.954
# --------------------------------
# ************Feature Freq=3********
# ------------label:0------------
# tp: 91, tn: 298, fp: 4, fn: 3
# accuracy:    0.982
# precison:    0.958
# recall:      0.968
# --------------------------------
# ------------label:1------------
# tp: 124, tn: 262, fp: 3, fn: 5
# accuracy:    0.980
# precison:    0.976
# recall:      0.961
# --------------------------------
# ------------label:-1------------
# tp: 124, tn: 277, fp: 2, fn: 1
# accuracy:    0.993
# precison:    0.984
# recall:      0.992
# --------------------------------
# ************Feature Freq=4********
# ------------label:0------------
# tp: 89, tn: 290, fp: 3, fn: 8
# accuracy:    0.972
# precison:    0.967
# recall:      0.918
# --------------------------------
# ------------label:1------------
# tp: 123, tn: 264, fp: 7, fn: 3
# accuracy:    0.975
# precison:    0.946
# recall:      0.976
# --------------------------------
# ------------label:-1------------
# tp: 122, tn: 278, fp: 5, fn: 4
# accuracy:    0.978
# precison:    0.961
# recall:      0.968
# --------------------------------
# ************Feature Freq=5********
# ------------label:0------------
# tp: 91, tn: 279, fp: 10, fn: 7
# accuracy:    0.956
# precison:    0.901
# recall:      0.929
# --------------------------------
# ------------label:1------------
# tp: 110, tn: 266, fp: 8, fn: 5
# accuracy:    0.967
# precison:    0.932
# recall:      0.957
# --------------------------------
# ------------label:-1------------
# tp: 122, tn: 276, fp: 2, fn: 8
# accuracy:    0.975
# precison:    0.984
# recall:      0.938
# --------------------------------
# ************Feature Freq=6********
# ------------label:0------------
# tp: 86, tn: 275, fp: 6, fn: 4
# accuracy:    0.973
# precison:    0.935
# recall:      0.956
# --------------------------------
# ------------label:1------------
# tp: 116, tn: 253, fp: 6, fn: 4
# accuracy:    0.974
# precison:    0.951
# recall:      0.967
# --------------------------------
# ------------label:-1------------
# tp: 112, tn: 284, fp: 3, fn: 7
# accuracy:    0.975
# precison:    0.974
# recall:      0.941
# --------------------------------
# ************Feature Freq=7********
# ------------label:0------------
# tp: 87, tn: 272, fp: 12, fn: 1
# accuracy:    0.965
# precison:    0.879
# recall:      0.989
# --------------------------------
# ------------label:1------------
# tp: 114, tn: 254, fp: 3, fn: 11
# accuracy:    0.963
# precison:    0.974
# recall:      0.912
# --------------------------------
# ------------label:-1------------
# tp: 104, tn: 277, fp: 4, fn: 7
# accuracy:    0.972
# precison:    0.963
# recall:      0.937
# --------------------------------
# ************Feature Freq=8********
# ------------label:0------------
# tp: 94, tn: 263, fp: 13, fn: 3
# accuracy:    0.957
# precison:    0.879
# recall:      0.969
# --------------------------------
# ------------label:1------------
# tp: 112, tn: 262, fp: 6, fn: 10
# accuracy:    0.959
# precison:    0.949
# recall:      0.918
# --------------------------------
# ------------label:-1------------
# tp: 106, tn: 285, fp: 5, fn: 11
# accuracy:    0.961
# precison:    0.955
# recall:      0.906
# --------------------------------
# ************Feature Freq=9********
# ------------label:0------------
# tp: 100, tn: 262, fp: 14, fn: 3
# accuracy:    0.955
# precison:    0.877
# recall:      0.971
# --------------------------------
# ------------label:1------------
# tp: 103, tn: 262, fp: 3, fn: 12
# accuracy:    0.961
# precison:    0.972
# recall:      0.896
# --------------------------------
# ------------label:-1------------
# tp: 103, tn: 280, fp: 6, fn: 8
# accuracy:    0.965
# precison:    0.945
# recall:      0.928
# --------------------------------
