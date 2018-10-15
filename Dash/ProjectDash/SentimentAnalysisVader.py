#imports
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#define analyzer
analyzer = SentimentIntensityAnalyzer()
'''
#sample try
#vs = analyzer.polarity_scores("VADER Sentiment looks interesting, I have high hopes!")
#print(vs)

check out vader sentiment github for details
{'neg': 0.0, 'neu': 0.463, 'pos': 0.537, 'compound': 0.6996}
'''

'''
pos_count = 0
pos_correct = 0

with open("positive.txt","r") as f:
    for line in f.read().split('\n'):
        vs = analyzer.polarity_scores(line)
        #pos
        if vs['compound'] > 0:
            pos_correct += 1
        pos_count +=1


neg_count = 0
neg_correct = 0

with open("negative.txt","r") as f:
    for line in f.read().split('\n'):
        vs = analyzer.polarity_scores(line)
        #neg
        if vs['compound'] <= 0:
            neg_correct += 1
        neg_count +=1

print("Positive accuracy = {}% via {} samples".format(pos_correct/pos_count*100.0, pos_count))
print("Negative accuracy = {}% via {} samples".format(neg_correct/neg_count*100.0, neg_count))

Positive accuracy = 69.4298574643661% via 5332 samples
Negative accuracy = 57.764441110277566% via 5332 samples
'''

'''
#using compound values from their docs to make it better

#let's go with the 0.5 and -0.5 as suggested by the documentation:
pos_count = 0
pos_correct = 0

threshold = 0.005

with open("positive.txt","r") as f:
    for line in f.read().split('\n'):
        vs = analyzer.polarity_scores(line)

        if vs['compound'] >= threshold or vs['compound'] <= -threshold:
            if vs['compound'] > 0:
                pos_correct += 1
            pos_count +=1


neg_count = 0
neg_correct = 0

with open("negative.txt","r") as f:
    for line in f.read().split('\n'):
        vs = analyzer.polarity_scores(line)
        if vs['compound'] >= threshold or vs['compound'] <= -threshold:
            if vs['compound'] <= 0:
                neg_correct += 1
            neg_count +=1

print("Positive accuracy = {}% via {} samples".format(pos_correct/pos_count*100.0, pos_count))
print("Negative accuracy = {}% via {} samples".format(neg_correct/neg_count*100.0, neg_count))

Positive accuracy = 87.22179585571757% via 2606 samples
Negative accuracy = 50.0% via 1818 samples
'''


'''
We threw out a lot of samples here, and we aren't doing much different
TextBlob. Should we give up? Maybe, but what if we instead look for
no conflict. So, what if we look only for signals where the opposite
is lower, or non-existent? For example, to classify something as positive 
why not require the neg bit to be less than 0.1 or something like:
'''


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import time
analyzer = SentimentIntensityAnalyzer()

pos_count = 0
pos_correct = 0

with open("positive.txt","r") as f:
    for line in f.read().split('\n'):
        vs = analyzer.polarity_scores(line)
        # if neg not more thna 0.1
        if not vs['neg'] > 0.1:
            # if pos -neg is geater than zero
            if vs['pos']-vs['neg'] > 0:
                pos_correct += 1
            pos_count +=1


neg_count = 0
neg_correct = 0

with open("negative.txt","r") as f:
    for line in f.read().split('\n'):
        vs = analyzer.polarity_scores(line)
        if not vs['pos'] > 0.1:
            if vs['pos']-vs['neg'] <= 0:
                neg_correct += 1
            neg_count +=1

print("Positive accuracy = {}% via {} samples".format(pos_correct/pos_count*100.0, pos_count))
print("Negative accuracy = {}% via {} samples".format(neg_correct/neg_count*100.0, neg_count))

'''
Positive accuracy = 80.69370058658507% via 3921 samples
Negative accuracy = 91.73643975245722% via 2747 samples
'''


'''
if sentiment was absolutely the *only* thing you planned to do with
this text, and you need it to be processed as fast as possible, 
then VADER sentiment is likely a better choice, 
going with that 0.05 threshdold which gave:
Positive accuracy = 99.85114617445669% via 3359 samples
Negative accuracy = 99.44954128440368% via 2180 samples
[Finished in 3.1s]

'''