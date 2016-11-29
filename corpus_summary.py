from my_tokenizer import *
from collections import Counter
import numpy
import random

PAD = "PAD"
LABELS = {-1 : 'negative', 0 : 'neutral', 1 : 'positive'}

def token_filter(function, tokens_map):
    return {k : tokens_map[k] for k in filter(function, tokens_map)}


def token_thresh_filter(tokens_map, threshold):
    return token_filter(lambda k : tokens_map[k] >= threshold, tokens_map)


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
    if k != 2:
        k = 2
    tf, t_doc_f = count_tokens(tweets)
    tf_thresh = token_thresh_filter(tf, thresh)
    t_doc_f_thresh = token_thresh_filter(t_doc_f, thresh)

    tf_bimers = Counter()
    t_doc_f_bimers = Counter()

    for tweet in tweets:
        tokens = tweet[0]
        padded_tweet = [PAD for i in xrange(k-1)] + tokens + [PAD for i in xrange(k-1)]
        bimers_set = set()
        for i in xrange(k-2, len(tokens)+k-1):
            left_gramer = padded_tweet[i:i+k-1]
            gramer = padded_tweet[i:i+k]
            right_gramer = padded_tweet[i+1:i+k]
            # make sure each gram meets the threshold
            if left_gramer in tf_thresh and right_gramer in tf_thresh:
                tf_bimers[gramer] += 1.0

            if left_gramer in tf_thresh and right_gramer in tf_thresh:
                if gramer not in bimers_set:
                    t_doc_f_bimers += 1.0
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
    for i, tok in tf_thresh:
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
