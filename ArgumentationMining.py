# -- coding: utf-8 --
"""
Created on Tue May 21 13:28:14 2019

@author: Meher
"""



import pandas as pd
import numpy as np
import nltk
from pandas import Series
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from   nltk.corpus import sentiwordnet as swn
from   nltk.corpus import stopwords
from   matplotlib import pyplot as plt



dataset = pd.read_csv('claim_stance_dataset_v1.csv')


train =[]
test = []
for index, row in dataset.iterrows():
    if row['split'] == "train":
         train.append((row[2:]))
    elif row['split'] == "test":
         test.append((row[2:]))

train = pd.DataFrame(train)
test = pd.DataFrame(test)
#

#Handling missing data
dataset = dataset[pd.notnull(dataset['claims.claimSentiment'])]

#Importing dataset
claim_corrected_data = dataset.iloc[:,7]
claim_target_data = dataset.iloc[:,16]
claim_sentiment_data = dataset.iloc[:,19]
topic_data = dataset.iloc[:,2]



claim = pd.DataFrame(claim_corrected_data)
claim['claim_target_data'] = claim_target_data

#relation_claim_topic = dataset.iloc[:,20] 

def similarity_feature (claim,evidence):
    return nltk.edit_distance(claim, evidence)

#Sentiment_feature
def sentiment_analyzer_scores(sentence):
    analyser = SentimentIntensityAnalyzer()
    score = analyser.polarity_scores(sentence)
    #print("{:-<40} {}".format(sentence, str(score)))
    if score['compound'] >= 0.05 : 
        print("Positive") 
        stance = +1
        print("stance",stance)
  
    elif score['compound'] <= - 0.05 : 
        print("Negative") 
        stance = -1
        print("stance",stance)
    else : 
        print("Neutral")
    return  score['neg'],score['neu'],score['pos'],score['compound']

def _sentiment_features(premise):
    analyser = SentimentIntensityAnalyzer()
    premise_words = nltk.word_tokenize(premise)
    num_of_positive_words = 0
    num_of_negative_words = 0
    num_of_neutral_words  = 0
    for word in premise_words:
        if analyser.polarity_scores(word)['neg'] > 0: 
            num_of_negative_words +=1
        if analyser.polarity_scores(word)['pos'] > 0: 
            num_of_positive_words +=1
        if analyser.polarity_scores(word)['neu'] > 0:
            num_of_neutral_words+=1
    return num_of_positive_words, num_of_negative_words , num_of_neutral_words  



# Number of words
def _num_of_words_feature(premise):
    premise_words = nltk.word_tokenize(premise)
    return len(premise_words)
#
for index, row in dataset.iterrows():
    x = sentiment_analyzer_scores(row['claims.claimCorrectedText'])

#num_of_positive_words, num_of_negative_words , num_of_neutral_words = _sentiment_features(claim_corrected_data[7])
# =============================================================================
# print("num_of_positive_words",num_of_positive_words)
# print("num_of_negative_words",num_of_negative_words)
# print("num_of_neutral_wordss",num_of_neutral_words)
# print("num of words",_num_of_words_feature(claim_corrected_data[7]))
# =============================================================================

def features(premise):
        # Number of words
        num_of_words_feature = _num_of_words_feature(premise)
        
        # Avg. Max. tfidf
        #avg_tfidf_feature, max_tfidf_feature = self._tfidf_features(premise)
        
        # positive score, nuetral score,  negative score, compound score
        negative_score,neutral_score,positive_score,compound_score = sentiment_analyzer_scores(premise)
        
        #similarity between topic and claim 
        #similarity_score = similarity_feature(premise, premises_text)
        # Number of postive/negative/neutral words
        num_of_positive_words,num_of_negative_words , num_of_neutral_words = _sentiment_features(premise)
        return [num_of_words_feature,
                #similarity_score,
                negative_score,
                neutral_score,
                positive_score,
                compound_score,
                num_of_positive_words, 
                num_of_negative_words, 
                num_of_neutral_words]
        

def _instance_features(premises):
    #premises_text = ' '.join(premises)
    premises_features = pd.DataFrame([features(premise) for premise in premises])
    return premises_features

_instance_features(claim_corrected_data[7])

#for index, row in dataset.iterrows():
#    _instance_features(row['claims.claimCorrectedText'])

