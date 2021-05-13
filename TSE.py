# -*- coding: utf-8 -*-
"""
Created on Sun May  9 13:57:47 2021

@author: Ayush
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('C:/Users/Ayush/OneDrive/Desktop/Twitter Sentiment analysis/Tweets.csv')
dataset.head()

dataset.isnull().values.any()
dataset=dataset.fillna(0)

print("Rows missing airline sentiment: {}".format(len(dataset.loc[dataset["airline_sentiment"]==0])))
print("Rows missing text: {}".format(len(dataset.loc[dataset["text"]==0])))
print("Rows missing negativereason confidence: {}".format(len(dataset.loc[dataset["negativereason_confidence"]==0])))

dataset.airline.value_counts().plot(kind='pie',autopct='%1.0f%%')

dataset_sentiment=dataset.groupby(['airline','airline_sentiment']).airline_sentiment.count().unstack()
dataset_sentiment.plot(kind='bar')

dataset.airline_sentiment.value_counts().plot(kind='pie',autopct='%1.0f%%')

dataset= dataset.drop(dataset[dataset['airline_sentiment'] == "neutral"].index)

sent_map={'positive':1,'negative':0}
dataset['airline_sentiment'] = dataset['airline_sentiment'].map(sent_map)
feature=dataset.iloc[:,10].values
label=dataset.iloc[:,1].values

import re
process_feature1=[]
for tweet in range(0,len(feature)):
    #filter the special character
    clean_tweet = re.sub(r'\W',' ',str(feature[tweet]))
    #filtering out the single characters
    clean_tweet = re.sub(r'\s+[a-zA-Z]\s+',' ',clean_tweet)
    #filtering out single character from the start
    clean_tweet = re.sub(r'\^[a-zA-Z]\s+',' ',clean_tweet)
    #subtituting multiple spaces wih single space
    clean_tweet = re.sub(r'\s+',' ',clean_tweet)
    #removing prefix b
    clean_tweet = re.sub(r'^b\s+','',clean_tweet)
    #lowing
    clean_tweet = clean_tweet.lower()
    clean_tweet = re.sub(r'^\s+','',clean_tweet)
    process_feature1.append(clean_tweet)

process_feature1 = np.array(process_feature1)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(process_feature1,label,test_size=0.3,random_state=40)

from sklearn.feature_extraction.text import TfidfVectorizer

sent_model= TfidfVectorizer(max_df=0.7,min_df=2,stop_words='english')

X_train = sent_model.fit_transform(X_train).toarray()
X_test = sent_model.transform(X_test).toarray()

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

nb= GaussianNB()
nb.fit(X_train,y_train)

pred_test = nb.predict(X_test)
print(accuracy_score(y_test,pred_test))


from sklearn.neighbors import KNeighborsClassifier
kneigh= KNeighborsClassifier(n_neighbors=5)
kneigh.fit(X_train,y_train)

pred = kneigh.predict(X_test)
print(accuracy_score(y_test,pred))


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100,random_state=40)
rf.fit(X_train,y_train)

pred_r = rf.predict(X_test)
print(accuracy_score(y_test,pred_r))


#save the model

import joblib
joblib.dump(sent_model,'sentiment-trained-tfidf.pkl')
joblib.dump(rf,'sentiment-rf-model.pkl')

# Check the model

feature1=['@VirginAmerica done! Thank you for the quick response, apparently faster than sitting on hold ;)']
process_feature1=[]
for tweet in range(0,len(feature1)):
    #filter the special character
    clean_tweet = re.sub(r'\W',' ',str(feature1[tweet]))
    #filtering out the single characters
    clean_tweet = re.sub(r'\s+[a-zA-Z]\s+',' ',clean_tweet)
    #filtering out single character from the start
    clean_tweet = re.sub(r'\^[a-zA-Z]\s+',' ',clean_tweet)
    #subtituting multiple spaces wih single space
    clean_tweet = re.sub(r'\s+',' ',clean_tweet)
    #removing prefix b
    clean_tweet = re.sub(r'^b\s+','',clean_tweet)
    #lowing
    clean_tweet = clean_tweet.lower()
    clean_tweet = re.sub(r'^\s+','',clean_tweet)
    process_feature1.append(clean_tweet)

process_feature1 = np.array(process_feature1)


X_twt = sent_model.transform(process_feature1).toarray()
pre = rf.predict(X_twt)
if pre[0]==1:
  print("Positive tweet")
else:
  print("Negative tweet")









