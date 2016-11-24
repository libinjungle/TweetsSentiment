
class TweetsProcessor(object):
    def map_tweets_candidates(self, filename):
        trump_file = open('./data/trump_tweets.txt', 'w')
        hillari_file = open('./data/hillari_tweets.txt', 'w')
        both_file = open('./data/both_tweets.txt', 'w')
        with open(filename) as tweets:
            for tweet in tweets:
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
                    both_file.write(tweet)
                    continue

                if hillari_flag:
                    hillari_file.write(tweet)

                if trump_flag:
                    trump_file.write(tweet)

        trump_file.close()
        hillari_file.close()
        both_file.close()


if __name__ == "__main__":
    tweets_corpus_file = "/Users/binli/PycharmProjects/TweetsSentiment/data/tweets_small_corpus"
    processor = TweetsProcessor()
    processor.map_tweets_candidates(tweets_corpus_file)




