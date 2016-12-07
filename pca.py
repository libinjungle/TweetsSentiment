from corpus_summary import *

LABELS = {-1 : 'negative', 0 : 'neutral', 1 : 'positive'}
DATA_FILE = "/Users/binli/PycharmProjects/TweetsSentiment/data/training.1600000.processed.noemoticon.csv"

class PCA(object):
    def __init__(self, k=400):
        self.E = None
        self.k = k

    def reduce(self, dataset):
        '''
        suppose n is the number of tweets, m is the number of features, pca reduce this n*m matrix
        to n*k matrix, k is the number of PCs.

        :param dataset: numpy array, the number of rows equals to the number of tweets, the number of
                        column equals to the number of features.
        :return:
        '''
        m = dataset.shape[0]
        n = dataset.shape[1]
        # 1*n
        mean = numpy.mean(dataset, axis=0)
        normalized_data = dataset - mean
        # n*n
        covar = numpy.cov(normalized_data.T)
        eigvals, eigvecs = numpy.linalg.eigh(covar)
        self.mean = mean
        self.covar = covar
        # 1*n
        self.eigvals = eigvals
        # n*n
        self.eigvecs = eigvals
        # get top k eigvals indices and select the top k eigen vectors
        # n*k
        self.E = eigvecs[:, (-eigvals).argsort()[0:self.k]]
        # m*k. So PCA transform m*n matrix to m*k matrix where k < n
        reduced_matrix = normalized_data.dot(self.E)
        return reduced_matrix


    def map(self, dataset):
        '''
        transform dataset to PCs
        :param dataset:
        :return:
        '''
        m = dataset.shape[0]
        normal_data = dataset - self.mean
        return normal_data.dot(self.E)


def generate_pca_data():
    '''
    transform original dataset(n*m) to n*k
    :return:
    '''
    tweets = construct_training_data(DATA_FILE)
    tf, t_doc_f = count_tokens(tweets)
    bi_f, bi_doc_f = count_bigrams(tweets, 2, 2)
    # 20, 30, 40 is reasonable dm for 4000 features
    for dm in [n * 10 for n in range(2, 4)]:
        for i in range(2, 4):
            token_map = create_gram_map(tf, i)
            bigram_map = create_gram_map(bi_f, i)

            uni_v = UnigramVectorizer(token_map)
            bigram_v = BigramVectorizer(bigram_map)
            vectorizers = [uni_v, bigram_v]
            pca = PCA(dm)
            for j, v in enumerate(vectorizers):
                # around 5000 tokens
                print('token size is %d' % v.tokens_size)
                if v.tokens_size < 20:
                    continue
                header = ",".join([str(k+1) for k in range(dm+1)])
                suffix = '.%d.%d.%d.csv' % (i, j, dm)
                dataset = generate_dataset_vectors(tweets, v)
                print('vector generated')
                tl, td = slicing_labels_features(dataset)
                # computation intensive step! needs to compute covariance matrix of n*m, n = 10000,
                # m = 5000
                t_red = pca.reduce(td)
                repacked = combine_labels_features(tl, t_red)
                print('ready to save.')
                numpy.savetxt("train"+suffix, repacked, fmt='%d'+',%.10f'*dm, delimiter=',', header=header)

if __name__ == "__main__":
    generate_pca_data()