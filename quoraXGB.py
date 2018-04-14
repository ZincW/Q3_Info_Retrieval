import numpy as np
import pandas as pd
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss
from nltk.corpus import stopwords
from collections import Counter
from sklearn.cross_validation import train_test_split
import math
import gensim
from gensim import corpora, models
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer

import xgboost as xgb

pal = sns.color_palette()

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)

stops = set(stopwords.words("english"))

def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    if math.isnan(R):
        R=0
    return R

train_word_match = df_train.apply(word_match_share, axis=1, raw=True)

def jaccard_word_match_share(row):
    wic = set(str(row['question1'])).intersection(set(str(row['question2'])))
    uw = set(str(row['question1'])).union(str(row['question2']))
    if len(uw) == 0:
        uw = [1]
    return (len(wic) / len(uw))

jaccard_train_word_match = df_train.apply(jaccard_word_match_share, axis=1, raw=True)

def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

eps = 5000
words = (" ".join(train_qs)).lower().split() #separate each letter with a space " " and split them into one single letter
counts = Counter(words) #count the frequency of each letter in words
weights = {word: get_weight(count) for word, count in counts.items()}


def tfidf_word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0

    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    R = np.sum(shared_weights) / np.sum(total_weights)
    if math.isnan(R):
        R=0
    return R

tfidf_train_word_match = df_train.apply(tfidf_word_match_share, axis=1, raw=True)

tokenizer = RegexpTokenizer(r'\w+')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

# list for tokenized documents in loop
texts = []

def LDA(row):
    # clean and tokenize document string
    Q1raw = str(row['question1']).lower().split()
    Q2raw = str(row['question2']).lower().split()

    tokens = Q1raw + Q2raw
    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if i not in stops]
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    # add tokens to list
    texts.append(stemmed_tokens)

LDA_data = df_train.apply(LDA, axis=1, raw=True)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# generate LDA model
ldamodel = Lda(corpus, num_topics=2, id2word=dictionary, passes=20)

# First we create our training and testing data
x_train = pd.DataFrame()
x_test = pd.DataFrame()
x_train['qid1'] = df_train['qid1']
x_train['qid2'] = df_train['qid2']
x_train['word_match'] = train_word_match
x_train['tfidf_word_match'] = tfidf_train_word_match
x_train['jaccard_word_match'] = jaccard_train_word_match
x_train['LDA_data'] = LDA_data
x_test['qid1'] = df_test['test_id']*2+1
x_test['qid2'] = df_test['test_id']*2+2
x_test['word_match'] = df_test.apply(word_match_share, axis=1, raw=True)
x_test['tfidf_word_match'] = df_test.apply(tfidf_word_match_share, axis=1, raw=True)
x_test['jaccard_word_match'] = df_test.apply(jaccard_word_match_share, axis=1, raw=True)
x_test['LDA'] = df_train.apply(LDA, axis=1, raw=True)

y_train = df_train['is_duplicate'].values

pos_train = x_train[y_train == 1]
neg_train = x_train[y_train == 0]

# Now we oversample the negative class
# There is likely a much more elegant way to do this...
p = 0.165
scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
while scale > 1:
    neg_train = pd.concat([neg_train, neg_train])
    scale -=1
neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])

x_train = pd.concat([pos_train, neg_train])
y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
del pos_train, neg_train

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

# Set our parameters for xgboost
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'error'
params['eta'] = 0.02 #Analogous to learing rate in GBM; default value is 0.3; It makes the model more robust by shrinking the weights on each step; Typical finaly values to be used: 0.01-0.2
params['max_depth'] = 4 #The maximum depth of a tree, same as GBM; Typical values: 3-10;
#used to control overfitting as higher depth will allow model to learn relations very specific to a particular sample

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)

d_test = xgb.DMatrix(x_test)
p_test = bst.predict(d_test)

sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = p_test
sub.to_csv('result.csv', index=False)
