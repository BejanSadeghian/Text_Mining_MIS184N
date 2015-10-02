# -*- coding: utf-8 -*-
"""
Created on Thu Oct 01 19:21:17 2015

Purpose: Homework 3 Part C
"""

import pandas as pd
import numpy as np
import math
from textblob import TextBlob
from nltk.corpus import stopwords
import re

rawdata = pd.read_csv(r'file:///C:\Users\beins_000\Documents\GitHub\Text_Mining_MIS184N\Group_Assignment_3\Assignment%203%20Edmunds%20Posts.csv')

rawdata = rawdata.ix[:,0]

N_before = 2
N_after = 5


stop = stopwords.words('english')
models = ['es', 'ls', 'rx', 'a8', 'a6', '3series', '5series', '7series', 'xj', 'sclass']
make = ['audi', 'bmw', 'jaguar', 'lexus', 'mercedes']

def getWords(text, stoplist = stop, makelist = make):
    #Gets the alpha numeric words and removed punctuations. Also checks stop words and make lists. Returns a list of words from text
    text1 = re.compile('\w+').findall(text)
    temp = [i.lower() for i in text1 if i.lower() not in stoplist]
    w1 = [i for i in temp if i not in makelist]
    return w1

rawdata.ix[0,0].split()
words_list = []
review_list = []
x=1
for nrow, row in enumerate(rawdata):
    
    words = getWords(row, stop, make)
    word_set = []
    if x == 1:    
        print words
        x = 0
    for n, word in enumerate(words):
        if word in models:
            #Perform polarity analysis on the string found            
            analyze_text = TextBlob(' '.join(words[(n-N_before):(n+N_after)]))
            #Assign the model found, the polarity, and the words to a tuple
            word_set.append((word.lower(), analyze_text.sentiment.polarity, words[(n-N_before):(n+N_after)]))
    if not not word_set:
        words_list.append(word_set)
        review_list.append(nrow)

dataframe = pd.DataFrame(index = range(len(review_list)), columns=models)
for nrow, row in enumerate(words_list):
    for model, senti, string in row:
        if not math.isnan(dataframe.ix[nrow, model]):
            dataframe.ix[nrow, model] = dataframe.ix[nrow, model] + senti
        else:
            dataframe.ix[nrow, model] = senti


dataframe.to_csv('take2senti.csv')