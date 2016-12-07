import time
import datetime
import random

class TweetsProcessor(object):
    def map_tweets_candidates(self, filename):
        '''
        aggregate tweet to its corresponding candidate. include candidate name and timestamp
        in the original tweet.

        :param filename:
        :return:
        '''
        trump_file = open('./data/trump_tweets_large.txt', 'w')
        hillari_file = open('./data/hillary_tweets_large.txt', 'w')
        both_file = open('./data/both_tweets_large.txt', 'w')
        count = 0
        tcount = 0
        hcount = 0
        with open(filename) as tweets:
            total_ts = time.time()
            for i, tweet in enumerate(tweets):
                if i != 0 and i%100 == 0:
                    print("processed %d tweets" % i)
                ts = time.time()
                str_ts = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                tweet = tweet.lower()
                hillari_flag = False
                trump_flag = False
                count += 1
                if 'hillari' in tweet \
                    or 'hillary' in tweet \
                    or 'clinto' in tweet:
                    hcount += 1
                    hillari_flag = True

                if 'donald' in tweet \
                    or 'trump' in tweet:
                    tcount += 1
                    trump_flag = True

                if hillari_flag and trump_flag:
                    both_file.write("hillari&trump" + "|||" + str_ts+'_'+str(count) + "|||" + tweet)
                    continue

                if hillari_flag:
                    hillari_file.write("hillari" + "|||" + str_ts+'_'+str(count) + "|||" + tweet)

                if trump_flag:
                    trump_file.write("trump" + "|||" + str_ts+'_'+str(count) + "|||" + tweet)

                time.sleep(0.1*random.random())
        print('total amount of tweets concerning trump: %d' % tcount)
        print('total amount of tweets concerning hillary: %d' % hcount)
        trump_file.close()
        hillari_file.close()
        both_file.close()


    def plain_tweets(self, output):
        with open(output, 'w') as out:
            with open('./data/hillary_tweets_large.txt') as trump_input:
                for line in trump_input:
                    tweet = line.split('|||')
                    out.write(tweet[2])




if __name__ == "__main__":
    # tweets_corpus_file = "/Users/binli/PycharmProjects/TweetsSentiment/data/tweets_small_corpus"
    tweets_corpus_file = "/Users/binli/PycharmProjects/TweetsSentiment/data/tweets_sample.txt"
    processor = TweetsProcessor()
    # processor.map_tweets_candidates(tweets_corpus_file)
    # processor.plain_tweets('./data/trump_plain_tweets_large.txt')
    processor.plain_tweets('./data/hillary_plain_tweets_large.txt')