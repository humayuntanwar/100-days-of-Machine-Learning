# Lexicon [chair, table,sppon, tv]

# I pulled the chair upto the table
# np.zeros(len(lexion))
# [0 0 0 0]
# [1 1 0 0]

import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
from nltk.tokenize import word_tokenize
# tokennizez the sentences into words
from nltk.stem import WordNetLemmatizer
# remove ings, and stuff
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 100000

#DAY 21
#read data in > stuff linto lexicon (do we care about any signle word?)
def create_lexicon(pos,neg):
    lexicon=[]
    for fi in [pos,neg]:
        with open(fi,'r') as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                all_words = word_tokenize(l.lower())
                lexicon+= list(all_words)
    
    #lemitize all words, stemming into legimitate words
    # input vector will be lexicon in length(short as possible wanted)
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon) # dict like elements
    #w_counts = {'the:'23, 'and': 344}
    l2 = []
    # we dont want common words like in, and , the words which occured more than 1000 and less than 50 are discarded
    for w in w_counts:
        if 1000 > w_counts[w] > 50:
            l2.append(w)
    print(len(l2))
    return l2

# func to classify feature set using lexicon 
# takes, sample, lexicon  , and what classfication gonna be
def sample_handling(sample, lexicon, classification):
    featureset=[]
    # 1, 0 pos
    #0,1 neg
    # [
    #     [],
    #     [1 0 1 1 0],[1,0]
    #     [],
    # ]
    # open what ever sample is 
    with open(sample, 'r') as f:
        contents = f.readlines()
        # for each of lines in our lines
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            # current word now lemitizer
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            # iterate through words ,a and set the index to 1 or plus 
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value]=+1
            features = list(features)
            featureset.append([features,classification])
    return featureset

# create feaures in sets , 10% test size
def create_feature_sets_and_labels(pos, neg,test_size=0.1):
    # create lexicon
    lexicon = create_lexicon(pos,neg) 
    features= [] #empty list
    features += sample_handling('pos.txt', lexicon,[1,0]) # our classification for pos is 1,0
    features += sample_handling('neg.txt', lexicon,[1,0]) # our calssification for neg is 0,1
    random.shuffle(features) # shuffle for NN
    features = np.array(features) # make features an array
    testing_size = int(test_size*len(features)) # whole number length of features
    #training data
    train_x = list(features[:,0][:-testing_size])  # all of zeroth elements
    train_y = list(features[:,0][:-testing_size])
    #testing data
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,0][-testing_size:])

    return train_x,train_y,test_x,test_y


if __name__ == '__main__':
    train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')
    with open('sentiment_set.pickle','wb') as f:
        pickle.dump([train_x,train_y,test_x,test_y],f)

