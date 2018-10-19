#using tweepy to interact with twitter
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sqlite3 #create database of tweets
from textblob import TextBlob
from unidecode import unidecode
import time


#database setup // twitter.db
conn = sqlite3.connect('twitter.db')
c = conn.cursor()

#creating a table,  unix = time, tweet= tweet, sentiment is vader
def create_table():
    try:
        c.execute("CREATE TABLE IF NOT EXISTS sentiment(unix REAL, tweet TEXT, sentiment REAL)")
        c.execute("CREATE INDEX fast_unix ON sentiment(unix)")
        c.execute("CREATE INDEX fast_tweet ON sentiment(tweet)")
        c.execute("CREATE INDEX fast_sentiment ON sentiment(sentiment)")
        conn.commit()
    except Exception as e:
        print(str(e))
create_table()

#apisecret, apikey, accesstoken, accesssecret loaded
ckey = open('apikey.txt','r').read()
csecret = open('apisecret.txt','r').read()
atoken = open('accesstoken.txt','r').read()
asecret = open('acesssecret.txt','r').read()

class listener(StreamListener):

    def on_data(self, data):
        try:
            data = json.loads(data)
            # only tweet text, not fancy emoji
            tweet = unidecode(data['text'])
            time_ms = data['timestamp_ms']
            #passing tweet to text blob
            analysis = TextBlob(tweet)
            #getting polarity -1...0..+1
            sentiment = analysis.sentiment.polarity
            print(time_ms, tweet, sentiment)
            # inserting nix, tweet, sentiment in db
            c.execute("INSERT INTO sentiment (unix, tweet, sentiment) VALUES (?, ?, ?)",
                (time_ms, tweet, sentiment))
            conn.commit()

        except KeyError as e:
            print(str(e))
        return(True)

    def on_error(self, status):
        print(status)


while True:
#try catch to keep recieivng tweet with sleep to allow blockage continuation
    try:
        auth = OAuthHandler(ckey, csecret)
        auth.set_access_token(atoken, asecret)
        twitterStream = Stream(auth, listener())
        twitterStream.filter(track=["a","e","i","o","u"])
    except Exception as e:
        print(str(e))
        time.sleep(5)