# -*- coding: utf-8 -*-
"""
Created on Wed Apr 08 19:47:54 2021

@author: silviya
"""


import pandas as pd
import numpy as np
import os

#_____Loaded csv data to pandas dataframe____(1)#
path = "C:/Users/Silviya/Documents/AI/final project"
filename = 'Youtube04-Eminem.csv'
fullpath = os.path.join(path,filename)

data = pd.read_csv(fullpath)    #Stored all data of given csv file to dataframe called 'data'

 
#_______Basic data Exploration_____(2)#
print(data.head(3))     #Diplaying first 3 records
print(data.shape)       #Displaying shape of dataframe

#Below, kept features that are important to train model and for prediction
#Removed 3 features(columns), COMMENT_ID, AUTHOR, DATE
#These 3 columns will not be helpful for trainning and prediction
to_keep = ['CONTENT','CLASS']
data_updated= data[to_keep]

train_x = data_updated['CONTENT']   #Stored feature X in variable train_x
train_y = data_updated['CLASS']     #Stored feature Y in varialbe train_y


#_____Preparing data for modeling_____(3)#
from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer()    #Created object of sklearn method 'CountVectorizer()'

#This will create vectors to count number of occurance of all unique words by removing stop words
CV_train_x= count_vectorizer.fit_transform(train_x) 

#_____Highlight of output_______(4)#
print(f"Shape of data after using CountVectorizer :{CV_train_x.shape}", )
print(CV_train_x)


#____Transform the data using tfidfTransform____(5)#
from sklearn.feature_extraction.text import TfidfTransformer

#tfid stands for Term Frequency(tf) and Inverse Document Frequency(idf)
#Tf total count of each word in a document(Here a comment) and divide it by total number of 
#words in that document(Here a same comment)

#idf check total number of documents(Here comments) that consist given word and divide it by 
#total number of all documents(Here all comments)

tfidf = TfidfTransformer()
tfidf_train_x= tfidf.fit_transform(CV_train_x)


print(type(tfidf_train_x))  #It will show type of data after transforming it by tfidf
print(tfidf_train_x.shape)  #Shape of data

#_____Shuffle data_____(6)#
dataframe_shuf = pd.DataFrame(tfidf_train_x.toarray(), columns= count_vectorizer.get_feature_names())
dataframe_shuf['CLASS'] = train_y
dataframe_shuf = dataframe_shuf.sample(frac = 1)


#_____Splited data into 75%-train and 25%- test without using train_test_split___(7)#

data_train = round(dataframe_shuf.shape[0] * (75/100))
x_data_train  = dataframe_shuf.iloc[:data_train, :-1]
y_data_train = dataframe_shuf.iloc[:data_train, -1]

x_data_test = dataframe_shuf.iloc[data_train:, :-1] 
y_data_test = dataframe_shuf.iloc[data_train :, -1]
#above, slice feature can also be used to split data. [start point: end point: selection]

#___Trained model using Naive Bayes____(8)#
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB().fit(x_data_train, y_data_train)


#_____Cross validated trainning data with 5 fold_____(9)#
from sklearn.model_selection import cross_val_score
scores = cross_val_score( MultinomialNB(), x_data_train, y_data_train, scoring='accuracy', cv=5)

print(f"Mean result of accuracy is: {scores.mean()}" ) #Display mean score of 5 folds
print(scores)   #display score of all 5 folds


#____Testing model by 25% data____#
y_data_pred = classifier.predict(x_data_test)  #Prediction by classifier(which is our model working on Naive Bayes classifier.)

#____Print accuracy and confusion matrix of model_____(10)#
from sklearn.metrics import confusion_matrix, accuracy_score

accuracy =  accuracy_score(y_data_test, y_data_pred)
print( f"Accuray of Model :{accuracy} ")

print("Confusion matrix :")
print(confusion_matrix(y_data_test,y_data_pred))





#___Added 6 new comments ___(11)#
new_comments =["Eminem doesn't try to keep up with the lyrics its the lyrics that try to keep up with him",
            'Nice song',
            'if this was an album, no doubt it would be his best',
            'Click on the link below and get 78 percent off on any electronics item',
            'Free shoes online application. Offer for limited period',
            'Eminen is the king of kings of all time his a Legendary Rapper'
         
            ]

#Transformed new comments
cv_new =  count_vectorizer.transform(new_comments)
tfidf_new = tfidf.transform(cv_new)
pred = classifier.predict(tfidf_new)

#__Compared Prediction____(12)#
predction_correct = np.sum((pred == [0, 0, 0, 1, 1, 0]))
print(f"Number of correct predictions: {predction_correct}" )




