#imports
from textblob import TextBlob

'''
pos_count = 0
pos_correct = 0
#open files read through
with open("positive.txt","r") as f:
    for line in f.read().split('\n'):
        analysis = TextBlob(line)
        #allows us to access sentiment polarity and subjectivity
        #if above 0 should be pos add to correct 
        if analysis.sentiment.polarity > 0:
            pos_correct += 1
        pos_count +=1


neg_count = 0
neg_correct = 0

with open("negative.txt","r") as f:
    for line in f.read().split('\n'):
        analysis = TextBlob(line)
        #allows us to access sentiment polarity and subjectivity
        #if less 0 should be pos add to negcorrect
        if analysis.sentiment.polarity <= 0:
            neg_correct += 1
        neg_count +=1

print("Positive accuracy = {}% via {} samples".format(pos_correct/pos_count*100.0, pos_count))
print("Negative accuracy = {}% via {} samples".format(neg_correct/neg_count*100.0, neg_count))


Positive accuracy = 71.11777944486121% via 5332 samples
Negative accuracy = 55.8702175543886% via 5332 samples
'''



'''
Alright! I can work with that. Recall the suggestion about -0.5 to 0.5
ing "neutral" with VADER?

What if we tried this with the TextBlob?
'''

pos_count = 0
pos_correct = 0

with open("positive.txt","r") as f:
    for line in f.read().split('\n'):
        analysis = TextBlob(line)

        if analysis.sentiment.polarity >= 0.001:
            if analysis.sentiment.polarity > 0:
                pos_correct += 1
            pos_count +=1


neg_count = 0
neg_correct = 0

with open("negative.txt","r") as f:
    for line in f.read().split('\n'):
        analysis = TextBlob(line)
        if analysis.sentiment.polarity <= -0.001:
            if analysis.sentiment.polarity <= 0:
                neg_correct += 1
            neg_count +=1

print("Positive accuracy = {}% via {} samples".format(pos_correct/pos_count*100.0, pos_count))
print("Negative accuracy = {}% via {} samples".format(neg_correct/neg_count*100.0, neg_count))
'''

Positive accuracy = 100.0% via 766 samples
Negative accuracy = 100.0% via 282 samples

lets change polarity to 0.001
Positive accuracy = 100.0% via 3790 samples
Negative accuracy = 100.0% via 2072 samples
'''