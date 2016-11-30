import numpy

PAD = 'PAD'

class Vectorizer(object):
    def __init__(self):
        pass

    def vectorize(self, tweet):
        raise NotImplementedError("subclasses must override vectorize()")


class UnigramVectorizer(Vectorizer):
    def __init__(self, token_map):
        # token_map, key is token, value is the index when constructing it
        # token_map = {'hello':0, 'bin':1, 'this':2}
        # the same as Vectorizer.__init__(self)
        super(UnigramVectorizer, self).__init__()
        self.token_map = token_map

    def vectorize(self, tweet, labeled=True):
        '''
        return numpy array with index as token's index and value as its occurrence
        :param tweet: list of word
        :return:
        '''
        v = numpy.zeros(self.tokens_size)
        if labeled:
            for token in tweet[0]:
                if token in self.token_map:
                    v[self.token_map[token]] += 1
            # did not consider those tokens that are not in map
        else:
            for token in tweet:
                if token in self.token_map:
                    v[self.token_map[token]] += 1
        return v

    @property
    def tokens_size(self):
        return len(self.token_map.keys())


class BigramVectorizer(Vectorizer):
    def __init__(self, bigram_map):
        super(BigramVectorizer, self).__init__()
        self.bigram_map = bigram_map

    def vectorize(self, tweet):
        v = numpy.zeros(self.tokens_size)
        padded = [PAD] + tweet[0] + [PAD]
        for i in range(0, len(padded)-1):
            bigram = tuple(padded[i:i+2])
            if bigram in self.bigram_map:
                v[self.bigram_map[bigram]] += 1
            # did not consider those bigrams that are not in map

        return v

    @property
    def tokens_size(self):
        return len(self.bigram_map.keys())



