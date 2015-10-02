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

rawsentiment = pd.read_csv(r'C:\Users\beins_000\Documents\GitHub\Text_Mining_MIS184N\Group_Assignment_3\Assignment 3 Sentiment Scores (2).csv')


#----------For Part C--------

rawsentiment1 = pd.read_csv(r'C:\Users\beins_000\Documents\GitHub\Text_Mining_MIS184N\Group_Assignment_3\take2senti.csv')
#rawsentiment = rawsentiment1.drop(['Unnamed: 0','model_strings'], axis=1)
rawsentiment = rawsentiment1.drop('Unnamed: 0', axis=1) #Testing bejans
lab_for_C = ['ES', 'LS', 'RX', 'A8', 'A6', '3series', '5series', '7series', 'XJ', 'Sclass']
rawsentiment.columns = lab_for_C
rawsentiment = rawsentiment * 5
rawsentiment.replace(to_replace=0,value=0.0001, inplace=True)

#----------End Part C--------

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
        node_list.append((pd.Series(averages_df.loc[:,'Node 1'])[index],pd.Series(averages_df.loc[:,'Node 2'])[index]))
        G.add_edge(*(opp_labels[pd.Series(averages_df.loc[:,'Node 1'])[index]],opp_labels[pd.Series(averages_df.loc[:,'Node 2'])[index]]))
for index, pref in enumerate(averages_df.loc[:,'Negative']): #Negatives
    if ~np.isnan(pref):
        node_list.append((pd.Series(averages_df.loc[:,'Node 2'])[index],pd.Series(averages_df.loc[:,'Node 1'])[index]))
        G.add_edge(*(opp_labels[pd.Series(averages_df.loc[:,'Node 2'])[index]],opp_labels[pd.Series(averages_df.loc[:,'Node 1'])[index]]))


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


##Part B

#Unweighted Page Rank
unweighted_PR = nx.pagerank(G, alpha = 0.5, max_iter=100)

#Weighted Page Rank
##Creating the network graph
FG=nx.Graph()

#Creating the labels for the nodes and adding a node to the G object
labels = {}
opp_labels = {}
for i, value in enumerate(names):
    labels[i] = str(value)
    opp_labels[value] = i
    FG.add_node(value) #Adding the nodes
#Create the list of tuples for edges with weights this time
node_list = []
for index, pref in enumerate(averages_df.loc[:,'Positive']): #Positives
    if ~np.isnan(pref):
        node_list.append((pd.Series(averages_df.loc[:,'Node 1'])[index],pd.Series(averages_df.loc[:,'Node 2'])[index],pref))
FG.add_weighted_edges_from(node_list)


for index, pref in enumerate(averages_df.loc[:,'Negative']): #Negatives
    if ~np.isnan(pref):
        node_list.append((pd.Series(averages_df.loc[:,'Node 2'])[index],pd.Series(averages_df.loc[:,'Node 1'])[index],pref))
FG.add_weighted_edges_from(node_list)


weighted_PR = nx.pagerank(FG, alpha = 0.5, weight = 'weight')#, max_iter=100000, tol=1e-01)


#Placing the PageRank in a series for combination and Correlation Calculation
i=0
unweighted_PageRank = pd.Series(name='Unweighted')
for dictkey in unweighted_PR:
    unweighted_PageRank[labels[i]] = unweighted_PR[dictkey]
    i = i + 1

weighted_PageRank = pd.Series(name='Weighted')
for dictkey in weighted_PR:
    weighted_PageRank[dictkey] = weighted_PR[dictkey]
    
#Create a dict of price data
price_dict = {'A6':20000, 'A8':12000, '3series':220000, '5series':60000, '7series':14000, 'XJ':6600, 'ES':135000, 'LS':30000, 'RX':120000, 'Sclass':25000}


price_list = pd.Series(name='Price')
for dictkey in price_dict:
    price_list[dictkey] = price_dict[dictkey]

price_pagerank = pd.concat([unweighted_PageRank, weighted_PageRank, price_list], axis=1)

#np.correlate(price_pagerank['Unweighted'], price_pagerank['Price'])

print('Unweighted Correlation to Price')
print(scipy.stats.spearmanr(price_pagerank['Unweighted'], price_pagerank['Price']))
print('Weighted Correlation to Price')
print(scipy.stats.spearmanr(price_pagerank['Weighted'], price_pagerank['Price']))


