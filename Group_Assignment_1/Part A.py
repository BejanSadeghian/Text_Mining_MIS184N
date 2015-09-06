# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 17:43:56 2015

@author: Bejan Sadeghian
"""

"""
Problems to Answer (Part A)
Part A (basic text mining)
A1. What are the top 5 parts of speech in this corpus of job descriptions? How frequently do they appear?
Hint: nltk.org is a great resource for exploring text mining with Python. There are many examples that are similar to the questions in this assignment.  

A2. Does this corpus support Zipf’s law? Plot the most common 100 words in the corpus against the theoretical prediction of the law. For this question, do not remove stopwords. Also do not perform stemming or lemmatization. 
Hint: Check http://www.garysieling.com/blog/exploring-zipfs-law-with-python-nltk-scipy-and-matplotlib 

A3. If we remove stopwords and lemmatize the corpus, what are the 10 most common words? What is their frequency?

"""

import pandas as pd
from nltk import word_tokenize
import nltk
import math
import matplotlib.pyplot as plt
from collections import Counter
from operator import itemgetter
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


#Import data and find unique tokens
rawdata = pd.read_csv(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Text Mining\Assignment 1\Train_rev1.csv')

desc = rawdata['FullDescription']

working_data = rawdata[0:5000]
wordslist = []


for index, row in enumerate(working_data['FullDescription']):
    string = row.decode('utf-8')
    temp = word_tokenize(string)
    for i in temp:
        wordslist.append(i)

wordset = list(set(wordslist))

#Lowercase the document
stemmed_tokens = [t.lower() for t in wordset if t.isalpha()]
lower_wordlist = [t.lower() for t in wordslist if t.isalpha()]
wordset = list(set(stemmed_tokens))

counts = Counter(lower_wordlist) #Returns a dict of words and their frequency in the corpus

#Parts of Speach
pos_tokens = nltk.pos_tag(wordset)

compiled_pos_freq = [(word, tag, counts[word]) for (word, tag) in pos_tokens]

#Create a frequency distribution
#list_of_tags = []
#for (word, tag, count) in compiled_pos_freq:
#    for i in range(0,count):
#        list_of_tags.append() #Compiled into the list comprension
tokens_fd = nltk.FreqDist(tag for (word, tag, count) in compiled_pos_freq for i in range(0,count)) #Cross check later to see if amount is correct

#tokens_fd1 = nltk.FreqDist(tag for (word, tag) in pos_tokens)

#Answers for A1
sorted_pos = sorted(tokens_fd.items(), key=itemgetter(1), reverse = True)
print 'Top 5 Parts of Speech (Question A1)'

xvar = [x for (x, y) in sorted_pos]
series_plot = pd.Series([y for (x, y) in sorted_pos], index=xvar)
print(series_plot[0:5].plot(kind='bar'))


#A2 requires calculation of a series of values for zipfs law, creating a two varaible dataframe
dict_wordcounts = Counter(lower_wordlist)
list_wordcounts = dict_wordcounts.most_common(100)

index = []
for i in range(1,len(list_wordcounts)+1):
    index.append(i)
top100 = pd.Series([b for (a, b) in list_wordcounts]) #Get the frequencies for the top 100 frequencies
top100words = pd.Series([a for (a, b) in list_wordcounts])  #The top freq words
#top100 = pd.Series([math.log(b) for (a, b) in sorted][0:101]) #Get the frequencies for the top 100 frequencies

zipf = []
for i,value in enumerate(top100):
    zipf.append((1000000.0/((i+1)*math.log(1.78*len(lower_wordlist))))) ##Scaling the Zipf's calculation due to difference in magnitude Unsure how to calculate Zipfs Law Prediction
index = pd.Series(index)
zipf = pd.Series(zipf)

zipf = pd.concat([index,zipf, top100], axis=1)

plt.plot(zipf[[0]], zipf[[1]], 'b-', label = 'Zipf Pred.')
plt.plot(zipf[[0]], zipf[[2]], 'r-', label = 'Actual')
plt.yscale('log')
plt.xlabel('Rank')
plt.ylabel('Token Frequency')
plt.xscale('log')
plt.legend()
plt.show()

print 'Does this corpus support Zipf\'s Law? (Question A2)'
print 'Zipf\'s law does hold true here because there is an inverse relationship between rank and token frequency, however comparing our predictured Zipfs value against our actual value showed some seperation likely due to stop words'

#For Part A3 we must remove stop words and lemmatize the corpus to find the most common parts of speech
#Remove Stop Words

#Lemmatize and remove stop words
nostop = [word for word in lower_wordlist if word not in stopwords.words('english')]
lmtzr = WordNetLemmatizer()
nostoplmtzed = [lmtzr.lemmatize(word) for word in nostop]
print('Top 10 Words without stop words and lemmatized, what are their frequencies? (Question A3)')
print(pd.Series(nostoplmtzed).value_counts()[:10])
print(pd.Series(nostoplmtzed).value_counts()[:10].plot(kind='bar'))