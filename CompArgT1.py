# -*- coding: utf-8 -*-
"""
Created on Sun May 26 00:18:43 2019

@author: Meher
"""

import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from   nltk.corpus import stopwords
from   matplotlib import pyplot as plt


# Import dataset
data = pd.read_csv('claim_stance_dataset_v1.csv')
# Handle missing data
dataset = data[pd.notnull(data['claims.claimSentiment'])]
#split traning and test data
train =[]
test = []
for index, row in dataset.iterrows():
    if row['split'] == "train":
         train.append((row[2:]))
    elif row['split'] == "test":
         test.append((row[2:]))

train = pd.DataFrame(train)
test = pd.DataFrame(test)



claim_corrected_data = dataset.iloc[:,7]
claim_target_data =dataset.iloc[:,16]
claim_sentiment_data = dataset.iloc[:,19]
topic_data = dataset.iloc[:,2]
topic_sentiment = dataset.iloc[:,4]
relation = dataset.iloc[:,20]
stance = dataset.iloc[:,6]


#tf_idf feature
def _tfidf_features(evidence):
        avg_tfidf_feature = 0
        max_tfidf_feature = 0
        premise_words = nltk.word_tokenize(evidence)
        tfidf = TfidfVectorizer()
# Whole dataset has to  be given for tfidf model
        tfidf_model = tfidf.fit(claim_corrected_data)
        tfidf_vector = tfidf_model.transform([evidence])
        avg_tfidf_feature = np.sum(tfidf_vector.toarray())/len(premise_words)
        max_tfidf_feature = np.max(tfidf_vector.toarray())
        return avg_tfidf_feature, max_tfidf_feature
# sentiment analyzer scores 
def sentiment_analyzer_scores(evidence):
    analyser = SentimentIntensityAnalyzer()
    score = analyser.polarity_scores(evidence)
    return  score['neg'],score['neu'],score['pos'],score['compound']

# sentiment analyzer for bag of words of negative/ positive/ nuetral
def _sentiment_analyzer_noOfWords(evidence):
    analyser = SentimentIntensityAnalyzer()
    premise_words = nltk.word_tokenize(evidence)
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
def _num_of_words_feature(evidence):
    premise_words = nltk.word_tokenize(evidence)
    return len(premise_words)


def feature_representation(evidence):
        # Number of words
        num_of_words_feature = _num_of_words_feature(evidence)
        
        # Avg. Max. tfidf
        avg_tfidf_feature, max_tfidf_feature = _tfidf_features(evidence)
        
        # positive score, nuetral score,  negative score, compound score
        negative_score,neutral_score,positive_score,compound_score = sentiment_analyzer_scores(evidence)
        
        # Number of postive/negative/neutral words
        num_of_positive_words, num_of_negative_words , num_of_neutral_words  = _sentiment_analyzer_noOfWords(evidence)

        return [num_of_words_feature,
                negative_score,
                neutral_score,
                positive_score,
                compound_score,
                num_of_positive_words, 
                num_of_negative_words, 
                num_of_neutral_words,
                avg_tfidf_feature, 
                max_tfidf_feature]
        

def _instance_features(evidences):
    evidence_features = pd.DataFrame([feature_representation(evidence) for evidence in evidences])
    return evidence_features

#Select columns from  train data
claim_corrected_data_train = train.iloc[:,5]
claim_target_data_train = train.iloc[:,14]
claim_sentiment_data_train = train.iloc[:,17]
stance_train = train.iloc[:,4]
#select columns from test data
claim_corrected_data_test = test.iloc[:,5]
claim_target_data_test = test.iloc[:,14]
claim_sentiment_data_test = test.iloc[:,17]
topic_sentiment_data_test = test.iloc[:,2]
claim_target_data_relation = test.iloc[:,18]
stance_test = test.iloc[:,4]
# claim sentiment from train data
X_train_class_data =  claim_sentiment_data_train

# Feautere calculations for the train data 
X_train_data =_instance_features(claim_corrected_data_train)


# Training SVM on features
from sklearn import svm
clf=svm.SVC(gamma='auto')
clf.fit(X_train_data,X_train_class_data)
Y_test_data = claim_corrected_data_test
Y_test_data_transfom =_instance_features(claim_corrected_data_test)
actual = claim_sentiment_data_test
Y_test_class_data = clf.predict(Y_test_data_transfom)

# Confusion Matrix for accuracy, precision, recall 
matrix = confusion_matrix(actual, Y_test_class_data)
print(matrix)

# Calculating precision, recall, fmeasure, accuracy from confusion matrix
accuray = (matrix[0][0]+matrix[1][1]) / (matrix[0][0]+matrix[1][1]+matrix[0][1]+matrix[1][0])
precision = matrix[1][1] / (matrix[0][1]+matrix[1][1])
recall    = matrix[0][0] / (matrix[0][0]+matrix[1][0])
f_measure = (2 * precision * recall ) / (precision + recall)
# Calculating stance 
predicted_stance =  topic_sentiment_data_test * claim_target_data_relation * Y_test_class_data


stance_filtered = stance_test.replace('PRO', 1)
stance_filtered = stance_filtered.replace('CON', -1)
# pro/ con to 1 / -1 
matrix = confusion_matrix(stance_filtered, predicted_stance)
print(matrix)

accuray_stance = (matrix[0][0]+matrix[1][1]) / 259
precision_stance = matrix[1][1] / (matrix[0][1]+matrix[1][1])
recall_stance    = matrix[0][0] / (matrix[0][0]+matrix[1][0])

f_measure_stance = (2 * precision * recall ) / (precision + recall)

predicted_stances = predicted_stance.replace('1', 'PRO')
predicted_stance = predicted_stances.replace('-1','CON')
predicted_stance = (predicted_stance)


x = pd.concat([pd.DataFrame(claim_corrected_data_test),pd.DataFrame(stance_test),pd.DataFrame(predicted_stance)])