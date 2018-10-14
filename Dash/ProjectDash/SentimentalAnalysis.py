# pip install textblob
#import nltk
#nltk.download('averaged_perceptron_tagger')
#test
from textblob import TextBlob
#from vaderSentiment.vaderSentiment import Sentimen

analysis = TextBlob("TextBlob sure looks like it has some interesting features!")
#To use something from TextBlob, we first want to convert it to a TextBlob object
print(dir(analysis))

# detect_language, capture noun_phrases,
#label parts of speech with tags, we can even translate to other languages, tokenize, and more
print(analysis.translate(to='es'))
#tags
print(analysis.tags)

#check out the sentiment:
#shows polarity and subjectivity
print(analysis.sentiment)

