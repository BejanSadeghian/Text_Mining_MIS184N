# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 17:50:24 2015

Purpose: Homework 3 Text Analytics Part A
"""
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy
import scipy.stats

from textblob import Word
from textblob.base import BaseTokenizer
from textblob import TextBlob
from textblob import Blobber
from textblob.tokenizers import WordTokenizer

from numpy import isnan
from nltk.corpus import stopwords

rawsentiment = pd.read_csv(r'Assignment 3 Sentiment Scores (2).csv')


########
#Part A#
########

#names = ['ES-LS','ES-RX','ES-A8','ES-A6','ES-3series','ES-5series','ES-7series','ES-XJ','ES-Sclass','LS-RX','LS-A8','LS-A6','LS-3series','LS-5series','LS-7series','LS-XJ','LS-Sclass','RX-A8','RX-A6','RX-3series','RX-5series','RX-7series','RX-XJ','RX-Sclass','A8-A6','A8-3series','A8-5series','A8-7series','A8-XJ','A8-Sclass','A6-3series','A6-5series','A6-7series','A6-XJ','A6-Sclass','3series-5series','3series-7series','3series-XJ','3series-Sclass','5series-7series','5series-XJ','5series-Sclass','7series-XJ','7series-Sclass','XJ-Sclass']
names = list(rawsentiment.columns)
#i=1
#j=1

##Create the sentiment difference dataframe (ES-LS, ES-RX, etc)
length_senti = len(rawsentiment)
width_senti = len(rawsentiment.T)
comparisons = pd.DataFrame(index=np.array(range(length_senti)))
#Vectors for future use in the nodes
node_name_1 = []
node_name_2 = []

n=1 #For counter
for i in range(width_senti-1):
    for j in range(width_senti - n):
        comparisons[str(names[i]) + '-' + str(names[(j+n)])] = np.array(rawsentiment[[i]]) - np.array(rawsentiment[[j+n]])
        node_name_1.append(names[i])
        node_name_2.append(names[(j+n)])
    n=n+1

length_comp = len(comparisons.T)

averages_df = pd.DataFrame(index=comparisons.columns, columns = ['Positive','Negative'])



for i in comparisons.columns:
    pos_sum = 0
    neg_sum = 0    
    pos_sum = [pos_sum + x for x in comparisons[i] if x > 0]
    neg_sum = [neg_sum + x for x in comparisons[i] if x < 0]
    if len(pos_sum) != 0:    
        averages_df.loc[i,'Positive'] = sum(pos_sum)/len(pos_sum)
    if len(neg_sum) != 0:
        averages_df.loc[i,'Negative'] = sum(neg_sum)/len(neg_sum)

averages_df['Node 1'] = node_name_1
averages_df['Node 2'] = node_name_2

##Creating the network graph
G=nx.DiGraph()

#Creating the labels for the nodes and adding a node to the G object
labels = {}
opp_labels = {}
for i, value in enumerate(names):
    labels[i] = str(value)
    opp_labels[value] = i
    G.add_node(i) #Adding the nodes
    

#Create the list of tuples for edges
node_list = []
for index, pref in enumerate(averages_df.loc[:,'Positive']): #Positives
    if ~np.isnan(pref):
        node_list.append((pd.Series(averages_df.loc[:,'Node 2'])[index],pd.Series(averages_df.loc[:,'Node 1'])[index]))
        G.add_edge(*(opp_labels[pd.Series(averages_df.loc[:,'Node 2'])[index]],opp_labels[pd.Series(averages_df.loc[:,'Node 1'])[index]]))
for index, pref in enumerate(averages_df.loc[:,'Negative']): #Negatives
    if ~np.isnan(pref):
        node_list.append((pd.Series(averages_df.loc[:,'Node 1'])[index],pd.Series(averages_df.loc[:,'Node 2'])[index]))
        G.add_edge(*(opp_labels[pd.Series(averages_df.loc[:,'Node 1'])[index]],opp_labels[pd.Series(averages_df.loc[:,'Node 2'])[index]]))


pos = nx.circular_layout(G) #Positions

#nx.edges([(1,2),(1,0)])#node_list)

nx.draw_networkx_labels(G,pos, labels, font_size=16)
#nx.draw_networkx_nodes(G,pos,node_color='skyblue')
nx.draw(G,pos,node_size=1000)
###nx.Graph(node_list)

fig1 = plt.gcf()
plt.figure(num=None, figsize=(30,20))
#plt.savefig('network.png', dpi=100)

plt.show()
fig1.savefig('network.png', dpi=1000)


########
#Part B#
########

##Coding pagerank ------- Weighted


#Weights out
wout_dict = {}
for index, value in enumerate(averages_df.ix[:,'Positive']):
    if ~np.isnan(value):
        if averages_df.ix[index,'Node 2'] in wout_dict:
            wout_dict[averages_df.ix[index,'Node 2']] = wout_dict[averages_df.ix[index,'Node 2']] + 1
        else:
            wout_dict[averages_df.ix[index,'Node 2']] = 1
for index, value in enumerate(averages_df.ix[:,'Negative']):
    if ~np.isnan(value):
        if averages_df.ix[index,'Node 1'] in wout_dict:
            wout_dict[averages_df.ix[index,'Node 1']] = wout_dict[averages_df.ix[index,'Node 1']] + 1
        else:
            wout_dict[averages_df.ix[index,'Node 1']] = 1

#Weights in
win_dict = {}
for index, value in enumerate(averages_df.ix[:,'Positive']):
    if ~np.isnan(value):
        if averages_df.ix[index,'Node 1'] in win_dict:
            win_dict[averages_df.ix[index,'Node 1']] = win_dict[averages_df.ix[index,'Node 1']] + 1
        else:
            win_dict[averages_df.ix[index,'Node 1']] = 1
for index, value in enumerate(averages_df.ix[:,'Negative']):
    if ~np.isnan(value):
        if averages_df.ix[index,'Node 2'] in win_dict:
            win_dict[averages_df.ix[index,'Node 2']] = win_dict[averages_df.ix[index,'Node 2']] + 1
        else:
            win_dict[averages_df.ix[index,'Node 2']] = 1

#Counting the number of outputs for each node
counts_dict = {}
#Create a dict of price data
price_dict = {'A6':20000, 'A8':12000, '3series':220000, '5series':60000, '7series':14000, 'XJ':6600, 'ES':135000, 'LS':30000, 'RX':120000, 'Sclass':25000}


price_list = pd.Series(name='Price')
for dictkey in price_dict:
    price_list[dictkey] = price_dict[dictkey]

for index, value in enumerate(averages_df.ix[:,'Negative']):
    if np.isnan(value):
        continue
    else:
        if averages_df.ix[index,'Node 1'] in counts_dict:
            counts_dict[averages_df.ix[index,'Node 1']] += 1
        else:
            counts_dict[averages_df.ix[index,'Node 1']] = 1


for index, value in enumerate(averages_df.ix[:,'Positive']):
    if np.isnan(value):
        continue
    else:
        if averages_df.ix[index,'Node 2'] in counts_dict:
            counts_dict[averages_df.ix[index,'Node 2']] += 1
        else:
            counts_dict[averages_df.ix[index,'Node 2']] = 1



#Creating the matrix
A = pd.DataFrame(index=names, columns=names)

for index, value in enumerate(averages_df.ix[:,'Positive']):
    if ~np.isnan(value):
        A.ix[averages_df.ix[index,'Node 1'], averages_df.ix[index,'Node 2']] = ((wout_dict[averages_df.ix[index,'Node 2']])*win_dict[averages_df.ix[index,'Node 2']]*value)
for index, value in enumerate(averages_df.ix[:,'Negative']):
    if ~np.isnan(value):
        A.ix[averages_df.ix[index,'Node 2'], averages_df.ix[index,'Node 1']] = ((wout_dict[averages_df.ix[index,'Node 1']])*win_dict[averages_df.ix[index,'Node 1']]*abs(value))

#weighting the values
vector_scalar = []
for row in range(len(A)):
    setofmakes = []
    for make, value in enumerate(A.ix[row,:]):
        if ~np.isnan(value):
            setofmakes.append(A.index[make])
    temp_win = 0
    temp_wout = 0
    for i in setofmakes:
        temp_win = win_dict[i] + temp_win
        temp_wout = wout_dict[i] + temp_wout
    vector_scalar.append((1.0/(temp_wout*temp_win)))

A = (A.T * vector_scalar).T


x = pd.DataFrame([1,1,1,1,1,1,1,1,1,1])

A.fillna(0, inplace=True)
new_names = list(A.index.values)
A = np.matrix(A) #Convert to Matrix
x = np.matrix(x) #Convert to Matrix


#Find Eigenvector/PageRank
iterations = 30

for i in range(iterations):
    new = A*x
    new = new/np.linalg.norm(new)
    if np.array_equal(np.array(x),np.array(new)):
        print 'Convergence after ', i
        break
    x = new

#print x
#print new

newseries = pd.DataFrame(new, index=new_names)

#Assign a index name to the pagerank
final = pd.concat([newseries,price_list], axis=1)
final.columns = ['PageRank','Sales'] 
print('Weighted Correlation')
print(scipy.stats.spearmanr(final['PageRank'], final['Sales']))
#print(scipy.stats.pearsonr(final['PageRank'], final['Price']))


##Coding pagerank ----- Unweighted

#Counting the number of outputs for each node
counts_dict = {}

for index, value in enumerate(averages_df.ix[:,'Negative']):
    if np.isnan(value):
        continue
    else:
        if averages_df.ix[index,'Node 1'] in counts_dict:
            counts_dict[averages_df.ix[index,'Node 1']] += 1
        else:
            counts_dict[averages_df.ix[index,'Node 1']] = 1


for index, value in enumerate(averages_df.ix[:,'Positive']):
    if np.isnan(value):
        continue
    else:
        if averages_df.ix[index,'Node 2'] in counts_dict:
            counts_dict[averages_df.ix[index,'Node 2']] += 1
        else:
            counts_dict[averages_df.ix[index,'Node 2']] = 1


#Creating the matrix
A = pd.DataFrame(index=names, columns=names)

for index, value in enumerate(averages_df.ix[:,'Positive']):
    if ~np.isnan(value):
        A.ix[averages_df.ix[index,'Node 1'], averages_df.ix[index,'Node 2']] = 1.0/counts_dict[averages_df.ix[index,'Node 2']]
for index, value in enumerate(averages_df.ix[:,'Negative']):
    if ~np.isnan(value):
        A.ix[averages_df.ix[index,'Node 2'], averages_df.ix[index,'Node 1']] = 1.0/counts_dict[averages_df.ix[index,'Node 1']]

x = pd.DataFrame([1,1,1,1,1,1,1,1,1,1])

A.fillna(0, inplace=True)
new_names = list(A.index.values)
A = np.matrix(A) #Convert to Matrix
x = np.matrix(x) #Convert to Matrix


#Find Eigenvector/PageRank
iterations = 30

for i in range(iterations):
    new = A*x
    new = new/np.linalg.norm(new)
    if np.array_equal(np.array(x),np.array(new)):
        print 'Convergence after ', i
        break
    x = new



newseries = pd.DataFrame(new, index=new_names)

#Assign a index name to the pagerank
final = pd.concat([final, newseries], axis=1)
final.columns = ['Weighted PageRank','Sales','Unweighted PageRank']
print('Unwighted Correlation')
print(scipy.stats.spearmanr(final['Unweighted PageRank'], final['Sales']))
print final









########
#Part C#
########


data=pd.read_csv("Assignment 3 Edmunds Posts.csv", usecols=[0])

data['Posts'] = data['Posts'].map(lambda x: x.replace("LexusES","ES"))
data['Posts'] = data['Posts'].map(lambda x: x.replace("ES330","ES"))
data['Posts'] = data['Posts'].map(lambda x: x.replace("LS460","LS"))
data['Posts'] = data['Posts'].map(lambda x: x.replace("LS470","LS"))
data['Posts'] = data['Posts'].map(lambda x: x.replace("LexusLS","LS"))
data['Posts'] = data['Posts'].map(lambda x: x.replace("LexusRX","RX"))

sw=stopwords.words('english')

sw.remove('not')

models_skinny=["es","ls","rx","a8","a6","3series","5series","7series","xj","sclass"] 

def tKnzr (xstring):
    global sw
    tokens=list(TextBlob(str(xstring)).words)
    removeStopWords=[word for word in tokens if word.lower() not in sw]
    lemmaed=[Word(w).lemmatize() for w in removeStopWords]
    lowercase=[word.lower() for word in lemmaed]
    return lowercase

def finder (tokensList, modelList, numberWords):
    blanklist=[]
    for i in xrange(len(tokensList)):
        if tokensList[i] in modelList:
            blanklist.append(tokensList[i-numberWords:i+1+numberWords])
    return blanklist
    
def sentiment_compiler(postseries,modelsList,numberofwords):
    tb=Blobber(tokenizer=WordTokenizer())
    
    newseries=postseries.apply(tKnzr)
    newseries=newseries.apply(lambda x: finder(x,modelsList, numberofwords))
    newseries=newseries.apply(lambda x: [' '.join(inner) for inner in x])
    
    models=[" "+x+" " for x in modelsList]
    df=pd.DataFrame(columns=modelsList).join(newseries, how="outer")
    
    for index,l in enumerate(newseries):
        for string in l: #looping over all the strings in model_strings
                for model, model_skinny in zip(models,modelsList): #loop over all the models in model list
                    if model in string: #check if model is in a particular list
                        if isnan(df[model_skinny].iloc[index]): #correcting for neutral
                            df[model_skinny].iloc[index]=0
                        df[model_skinny].iloc[index]+=tb(string).sentiment[0]  
    return df
    
    


rawsentiment1 = sentiment_compiler(data['Posts'],models_skinny,2)
rawsentiment = rawsentiment1.drop(['Posts'], axis = 1)

#For testing take2senti file
#rawsentiment1 = pd.read_csv(r'C:\Users\beins_000\Documents\GitHub\Text_Mining_MIS184N\Group_Assignment_3\take2senti.csv')
#rawsentiment = rawsentiment1.drop('Unnamed: 0', axis=1) #Testing bejans

lab_for_C = ['ES', 'LS', 'RX', 'A8', 'A6', '3series', '5series', '7series', 'XJ', 'Sclass']
rawsentiment.columns = lab_for_C
rawsentiment = rawsentiment * 5


#names = ['ES-LS','ES-RX','ES-A8','ES-A6','ES-3series','ES-5series','ES-7series','ES-XJ','ES-Sclass','LS-RX','LS-A8','LS-A6','LS-3series','LS-5series','LS-7series','LS-XJ','LS-Sclass','RX-A8','RX-A6','RX-3series','RX-5series','RX-7series','RX-XJ','RX-Sclass','A8-A6','A8-3series','A8-5series','A8-7series','A8-XJ','A8-Sclass','A6-3series','A6-5series','A6-7series','A6-XJ','A6-Sclass','3series-5series','3series-7series','3series-XJ','3series-Sclass','5series-7series','5series-XJ','5series-Sclass','7series-XJ','7series-Sclass','XJ-Sclass']
names = list(rawsentiment.columns)
#i=1
#j=1

##Create the sentiment difference dataframe (ES-LS, ES-RX, etc)
length_senti = len(rawsentiment)
width_senti = len(rawsentiment.T)
comparisons = pd.DataFrame(index=np.array(range(length_senti)))
#Vectors for future use in the nodes
node_name_1 = []
node_name_2 = []

n=1 #For counter
for i in range(width_senti-1):
    for j in range(width_senti - n):
        comparisons[str(names[i]) + '-' + str(names[(j+n)])] = np.array(rawsentiment[[i]]) - np.array(rawsentiment[[j+n]])
        node_name_1.append(names[i])
        node_name_2.append(names[(j+n)])
    n=n+1

length_comp = len(comparisons.T)

averages_df = pd.DataFrame(index=comparisons.columns, columns = ['Positive','Negative'])



for i in comparisons.columns:
    pos_sum = 0
    neg_sum = 0    
    pos_sum = [pos_sum + x for x in comparisons[i] if x > 0]
    neg_sum = [neg_sum + x for x in comparisons[i] if x < 0]
    if len(pos_sum) != 0:    
        averages_df.loc[i,'Positive'] = sum(pos_sum)/len(pos_sum)
    if len(neg_sum) != 0:
        averages_df.loc[i,'Negative'] = sum(neg_sum)/len(neg_sum)

averages_df['Node 1'] = node_name_1
averages_df['Node 2'] = node_name_2

##Creating the network graph
G=nx.DiGraph()

#Creating the labels for the nodes and adding a node to the G object
labels = {}
opp_labels = {}
for i, value in enumerate(names):
    labels[i] = str(value)
    opp_labels[value] = i
    G.add_node(i) #Adding the nodes
    

#Create the list of tuples for edges
node_list = []
for index, pref in enumerate(averages_df.loc[:,'Positive']): #Positives
    if ~np.isnan(pref):
        node_list.append((pd.Series(averages_df.loc[:,'Node 2'])[index],pd.Series(averages_df.loc[:,'Node 1'])[index]))
        G.add_edge(*(opp_labels[pd.Series(averages_df.loc[:,'Node 2'])[index]],opp_labels[pd.Series(averages_df.loc[:,'Node 1'])[index]]))
for index, pref in enumerate(averages_df.loc[:,'Negative']): #Negatives
    if ~np.isnan(pref):
        node_list.append((pd.Series(averages_df.loc[:,'Node 1'])[index],pd.Series(averages_df.loc[:,'Node 2'])[index]))
        G.add_edge(*(opp_labels[pd.Series(averages_df.loc[:,'Node 1'])[index]],opp_labels[pd.Series(averages_df.loc[:,'Node 2'])[index]]))


pos = nx.circular_layout(G) #Positions

#nx.edges([(1,2),(1,0)])#node_list)

nx.draw_networkx_labels(G,pos, labels, font_size=16)
#nx.draw_networkx_nodes(G,pos,node_color='skyblue')
nx.draw(G,pos,node_size=1000)
###nx.Graph(node_list)

fig1 = plt.gcf()
plt.figure(num=None, figsize=(30,20))
#plt.savefig('network.png', dpi=100)

plt.show()
fig1.savefig('network.png', dpi=1000)


##Coding pagerank ------- Weighted


#Weights out
wout_dict = {}
for index, value in enumerate(averages_df.ix[:,'Positive']):
    if ~np.isnan(value):
        if averages_df.ix[index,'Node 2'] in wout_dict:
            wout_dict[averages_df.ix[index,'Node 2']] = wout_dict[averages_df.ix[index,'Node 2']] + 1
        else:
            wout_dict[averages_df.ix[index,'Node 2']] = 1
for index, value in enumerate(averages_df.ix[:,'Negative']):
    if ~np.isnan(value):
        if averages_df.ix[index,'Node 1'] in wout_dict:
            wout_dict[averages_df.ix[index,'Node 1']] = wout_dict[averages_df.ix[index,'Node 1']] + 1
        else:
            wout_dict[averages_df.ix[index,'Node 1']] = 1

#Weights in
win_dict = {}
for index, value in enumerate(averages_df.ix[:,'Positive']):
    if ~np.isnan(value):
        if averages_df.ix[index,'Node 1'] in win_dict:
            win_dict[averages_df.ix[index,'Node 1']] = win_dict[averages_df.ix[index,'Node 1']] + 1
        else:
            win_dict[averages_df.ix[index,'Node 1']] = 1
for index, value in enumerate(averages_df.ix[:,'Negative']):
    if ~np.isnan(value):
        if averages_df.ix[index,'Node 2'] in win_dict:
            win_dict[averages_df.ix[index,'Node 2']] = win_dict[averages_df.ix[index,'Node 2']] + 1
        else:
            win_dict[averages_df.ix[index,'Node 2']] = 1

#Counting the number of outputs for each node
counts_dict = {}
#Create a dict of price data
price_dict = {'A6':20000, 'A8':12000, '3series':220000, '5series':60000, '7series':14000, 'XJ':6600, 'ES':135000, 'LS':30000, 'RX':120000, 'Sclass':25000}


price_list = pd.Series(name='Price')
for dictkey in price_dict:
    price_list[dictkey] = price_dict[dictkey]

for index, value in enumerate(averages_df.ix[:,'Negative']):
    if np.isnan(value):
        continue
    else:
        if averages_df.ix[index,'Node 1'] in counts_dict:
            counts_dict[averages_df.ix[index,'Node 1']] += 1
        else:
            counts_dict[averages_df.ix[index,'Node 1']] = 1


for index, value in enumerate(averages_df.ix[:,'Positive']):
    if np.isnan(value):
        continue
    else:
        if averages_df.ix[index,'Node 2'] in counts_dict:
            counts_dict[averages_df.ix[index,'Node 2']] += 1
        else:
            counts_dict[averages_df.ix[index,'Node 2']] = 1



#Creating the matrix
A = pd.DataFrame(index=names, columns=names)

for index, value in enumerate(averages_df.ix[:,'Positive']):
    if ~np.isnan(value):
        A.ix[averages_df.ix[index,'Node 1'], averages_df.ix[index,'Node 2']] = ((wout_dict[averages_df.ix[index,'Node 2']])*win_dict[averages_df.ix[index,'Node 2']]*value)
for index, value in enumerate(averages_df.ix[:,'Negative']):
    if ~np.isnan(value):
        A.ix[averages_df.ix[index,'Node 2'], averages_df.ix[index,'Node 1']] = ((wout_dict[averages_df.ix[index,'Node 1']])*win_dict[averages_df.ix[index,'Node 1']]*abs(value))

#weighting the values
vector_scalar = []
for row in range(len(A)):
    setofmakes = []
    for make, value in enumerate(A.ix[row,:]):
        if ~np.isnan(value):
            setofmakes.append(A.index[make])
    temp_win = 0
    temp_wout = 0
    for i in setofmakes:
        temp_win = win_dict[i] + temp_win
        temp_wout = wout_dict[i] + temp_wout
    vector_scalar.append((1.0/(temp_wout*temp_win)))

A = (A.T * vector_scalar).T


x = pd.DataFrame([1,1,1,1,1,1,1,1,1,1])

A.fillna(0, inplace=True)
new_names = list(A.index.values)
A = np.matrix(A) #Convert to Matrix
x = np.matrix(x) #Convert to Matrix


#Find Eigenvector/PageRank
iterations = 30

for i in range(iterations):
    new = A*x
    new = new/np.linalg.norm(new)
    if np.array_equal(np.array(x),np.array(new)):
        print 'Convergence after ', i
        break
    x = new

#print x
#print new

newseries = pd.DataFrame(new, index=new_names)

#Assign a index name to the pagerank
final = pd.concat([newseries,price_list], axis=1)
final.columns = ['PageRank','Sales'] 
print('Weighted Correlation')
print(scipy.stats.spearmanr(final['PageRank'], final['Sales']))
#print(scipy.stats.pearsonr(final['PageRank'], final['Price']))


##Coding pagerank ----- Unweighted

#Counting the number of outputs for each node
counts_dict = {}

for index, value in enumerate(averages_df.ix[:,'Negative']):
    if np.isnan(value):
        continue
    else:
        if averages_df.ix[index,'Node 1'] in counts_dict:
            counts_dict[averages_df.ix[index,'Node 1']] += 1
        else:
            counts_dict[averages_df.ix[index,'Node 1']] = 1


for index, value in enumerate(averages_df.ix[:,'Positive']):
    if np.isnan(value):
        continue
    else:
        if averages_df.ix[index,'Node 2'] in counts_dict:
            counts_dict[averages_df.ix[index,'Node 2']] += 1
        else:
            counts_dict[averages_df.ix[index,'Node 2']] = 1


#Creating the matrix
A = pd.DataFrame(index=names, columns=names)

for index, value in enumerate(averages_df.ix[:,'Positive']):
    if ~np.isnan(value):
        A.ix[averages_df.ix[index,'Node 1'], averages_df.ix[index,'Node 2']] = 1.0/counts_dict[averages_df.ix[index,'Node 2']]
for index, value in enumerate(averages_df.ix[:,'Negative']):
    if ~np.isnan(value):
        A.ix[averages_df.ix[index,'Node 2'], averages_df.ix[index,'Node 1']] = 1.0/counts_dict[averages_df.ix[index,'Node 1']]

x = pd.DataFrame([1,1,1,1,1,1,1,1,1,1])

A.fillna(0, inplace=True)
new_names = list(A.index.values)
A = np.matrix(A) #Convert to Matrix
x = np.matrix(x) #Convert to Matrix


#Find Eigenvector/PageRank
iterations = 30

for i in range(iterations):
    new = A*x
    new = new/np.linalg.norm(new)
    if np.array_equal(np.array(x),np.array(new)):
        print 'Convergence after ', i
        break
    x = new



newseries = pd.DataFrame(new, index=new_names)

#Assign a index name to the pagerank
final = pd.concat([final, newseries], axis=1)
final.columns = ['Weighted PageRank','Sales','Unweighted PageRank']
print('Unwighted Correlation')
print(scipy.stats.spearmanr(final['Unweighted PageRank'], final['Sales']))
print final







