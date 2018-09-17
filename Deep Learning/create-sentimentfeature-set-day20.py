# Lexicon [chair, table,sppon, tv]

# I pulled the chair upto the table
# np.zeros(len(lexion))
# [0 0 0 0]
# [1 1 0 0]

import nltk
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


