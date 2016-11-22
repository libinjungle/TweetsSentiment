from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import json

# Variables that contains the user credentials to access Twitter API
access_token = ''
access_token_secret = ''
consumer_key = ''
consumer_secret = ''


class StdOutListener(StreamListener):
    def on_data(self, data):
        json_load = json.loads(data)
        key = 'text'
        if key in json_load:
            texts = json_load['text']
            coded = texts.encode('ascii', 'ignore')
            s = str(coded)
            myfile = open('tweets_stream.txt', 'a')
            myfile.write(s[:])
            myfile.write('\n')
            myfile.flush()
            # myfile.write('\n')
            myfile.close()
        return True

    def on_error(self, status):
        if status == 420:
            return False
        print(status)

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
stream = Stream(auth, StdOutListener())

# This line filter Twitter Streams to capture data by the keywords: 'python', 'javascript', 'ruby'
stream.filter(track=['donald', 'trump', 'hillary', 'clinton'], languages=['en'])
