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
        trump_file = open('./data/trump_tweets.txt', 'w')
        hillari_file = open('./data/hillari_tweets.txt', 'w')
        both_file = open('./data/both_tweets.txt', 'w')
        count = 1
        with open(filename) as tweets:
            total_ts = time.time()
            for tweet in tweets:
                ts = time.time()
                str_ts = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                tweet = tweet.lower()
                hillari_flag = False
                trump_flag = False
                if 'hillari' in tweet \
                    or 'hillary' in tweet \
                    or 'clinto' in tweet:
                    hillari_flag = True

                if 'donald' in tweet \
                    or 'trump' in tweet:
                    trump_flag = True

                if hillari_flag and trump_flag:
                    both_file.write("hillari&trump" + "|||" + str_ts+'_'+str(count) + "|||" + tweet)
                    continue

                if hillari_flag:
                    hillari_file.write("hillari" + "|||" + str_ts+'_'+str(count) + "|||" + tweet)

                if trump_flag:
                    trump_file.write("trump" + "|||" + str_ts+'_'+str(count) + "|||" + tweet)

                count += 1
                time.sleep(0.1*random.random())

        trump_file.close()
        hillari_file.close()
        both_file.close()


if __name__ == "__main__":
    tweets_corpus_file = "/Users/binli/PycharmProjects/TweetsSentiment/data/tweets_small_corpus"
    processor = TweetsProcessor()
    processor.map_tweets_candidates(tweets_corpus_file)




