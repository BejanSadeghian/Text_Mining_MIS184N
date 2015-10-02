# -*- coding: utf-8 -*-
"""
Created on Thu Oct 01 19:21:17 2015

Purpose: Homework 3 Part C
"""

import pandas as pd
from textblob import TextBlob

rawdata = pd.read_csv(r'file:///C:\Users\beins_000\Documents\GitHub\Text_Mining_MIS184N\Group_Assignment_3\Assignment%203%20Edmunds%20Posts.csv')

rawdata = rawdata.ix[:,0]

N_before = 3
N_after = 3

models = ['es', 'ls', 'rx', 'a8', 'a6', '3series', '5series', '7series', 'xj', 'sclass']
make = ['audi', 'bmw', 'jaguar', 'lexus', 'mercedes']

rawdata.ix[0,0].split()
words_list = []
review_list = []
for nrow, row in enumerate(rawdata):
    words = row.split()
    word_set = []
    for n, word in enumerate(words):            
        if word.lower() in models:
            word_set.append(words[(n-N_before):(n+N_after)])
    if not not word_set:
        words_list.append(word_set)
        review_list.append(nrow)
DF_words = pd.concat([pd.Series(review_list),pd.Series(words_list)], axis=1)

for wordlist in DF_words[1]:
    for string in wordlist:
        analyze_text = TextBlob(' '.join(string))
        print analyze_text.sentiment.polarity