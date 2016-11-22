import sys
import os
import time
import tweepy

API_KEY = ''
API_SECRET = ''

auth = tweepy.AppAuthHandler(API_KEY, API_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
if not api:
#  print("Can not make authentication to API.")
  sys.exit(-1)


searchQuery = 'donald+OR+trump+OR+hillary+OR+clinton'  # this is what we're searching for
maxTweets = 100000000 # Some arbitrary large number
tweetsPerQry = 100  # this is the max the API permits
fName = 'tweets.txt' # We'll store the tweets in a text file.


# If results from a specific ID onwards are reqd, set since_id to that ID.
# else default to no lower limit, go as far back as API allows
sinceId = None

# If results only below a specific ID are, set max_id to that ID.
# else default to no upper limit, start from the most recent tweet matching the search query.
max_id = -1L

tweetCount = 0
#print("Downloading max {0} tweets".format(maxTweets))
with open(fName, 'w') as f:
    while tweetCount < maxTweets:
        time.sleep(3)
        try:
            if (max_id <= 0):
                if (not sinceId):
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry)
                else:
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                            since_id=sinceId)
            else:
                if (not sinceId):
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                            max_id=str(max_id - 1))
                else:
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                            max_id=str(max_id - 1),
                                            since_id=sinceId)
            if not new_tweets:
                #print("No more tweets found")
                break
            for tweet_raw in new_tweets:
                tweet = tweet_raw._json # tweet is dict
                if 'text' in tweet:
                  text = tweet['text']
                  coded = text.encode('ascii', 'ignore')
                  s = str(coded)
                  f.write(s + '\n')
                  f.flush()

            tweetCount += len(new_tweets)
            print("Downloaded {0} tweets".format(tweetCount))
            max_id = new_tweets[-1].id
        except tweepy.TweepError as e:
            # Just exit if any error
            print("some error : " + str(e))
            print("Sleep three minutes...")
            time.sleep(180)
            continue

print ("Downloaded {0} tweets, Saved to {1}".format(tweetCount, fName))