import numpy

class Classifier(object):
    def __init__(self):
        pass

    def train(self, training_data, labels):
        raise NotImplementedError("Subclasses must override train()")
        pass

    def classify(self, tweet):
        raise NotImplementedError("Subclasses must override classify()")
        pass

    def classify_tweets(self, tweets):
        ans = numpy.zeros(len(tweets))
        for i, t in enumerate(tweets):
            ans[i] = self.classify(t)
            # print(ans[i])
        return ans


class NaiveBayesClassifier(Classifier):
    def __init__(self, n, labels):
        '''
        classifier will be used for all runs of k-fold validation
        :param n: feature size
        :param labels: [-1 : 'negative', 0 : 'neutral', 1 : 'positive']
        '''
        super(NaiveBayesClassifier, self).__init__()
        # Note: these global variables are updated in different runs of validations.
        self.n = n
        self.label_num = len(labels)
        self.label_feature_matrix = None
        self.label_to_idx = {l : i for i, l in enumerate(labels)}
        self.idx_to_label = {i : l for i, l in enumerate(labels)}
        self.features_prior = None
        self.labels_prior = None


    def train(self, training_data, labels_col):
        '''
        the goal is to get three priors for following classification.
        1. class_priors
           ------------
           the probability of each label
        2. feature_priors
           --------------
           the probability of each feature in all feature space
        3. theta
           -----
           the probability of each feature under certain label

        :param training_set: numpy array, k rows, n cols, k is the number of tweet, n is the number
                             of features
        :param labels: numpy array, k rows, 1 col, k is the number of tweet, the element is the
                        label of this tweet
        :return:
        '''
        labels_prior = numpy.zeros([self.label_num, 1])
        features_prior = numpy.zeros([1, self.n])
        # shape[0] gives the number of rows
        num_samples = labels_col.shape[0]
        label_feature_matrix = numpy.zeros([self.label_num, self.n])
        for label in self.label_to_idx:
            print('label is: %d ' % label)
            indexes = numpy.where(labels_col == label)[0] # 0 means row indices
            sliced_samples = training_data[indexes, :]
            idx = self.label_to_idx[label]
            print('index is: %d' % idx)
            label_feature_matrix[idx, :] = sliced_samples.sum(0) + 2
            features_prior += label_feature_matrix[idx, :]
            label_feature_matrix[idx, :] *= 1.0 / label_feature_matrix[idx, :].sum()
            labels_prior[idx] = len(indexes) / float(num_samples)
        features_prior *= 1.0 / features_prior.sum()
        self.features_prior = numpy.log(features_prior).T
        self.label_feature_matrix = numpy.log(label_feature_matrix).T
        self.labels_prior = numpy.log(labels_prior)


    def classify(self, tweet):
        '''
        :param tweet: tweet that has cut labels
        :return:
        '''
        sample = tweet.reshape([1, self.n])
        # 1-D array
        post = sample.dot(self.label_feature_matrix) - sample.dot(self.features_prior) + self.labels_prior.T
        return self.idx_to_label[numpy.argmax(post)]






