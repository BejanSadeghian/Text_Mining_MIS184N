# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 17:50:24 2015

Purpose: Homework 3 Text Analytics Part A
"""
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math


rawsentiment = pd.read_csv(r'C:\Users\beins_000\Documents\GitHub\Text_Mining_MIS184N\Group_Assignment_3\Assignment 3 Sentiment Scores (2).csv')

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
unweighted_PR = nx.pagerank(G, alpha = 0.5)

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


weighted_PR = nx.pagerank(FG, alpha = 0.5, weight = 'weight')












#PR_ES = 1
#PR_LS = 1
#PR_RX =1
#PR_A8 =1
#PR_A6 = 1
#PR_3series = 1
#PR_5series = 1
#PR_7series = 1
#PR_XJ = 1
#PR_Sclass = 1
#
##Count the number of 
#ref = {}
#for n in names:
#    count = int(0)
#    count = [(count + 1) for (y, x) in node_list if x == n]
#    ref[n] = sum(count)
#RF_ES = float(1)/ref['ES']
#RF_LS = float(1)/ref['LS']
#RF_RX =float(1)/ref['RX']
#RF_A8 =float(1)/ref['A8']
#RF_A6 = float(1)/ref['A6']
#RF_3series = float(1)/ref['3series']
#RF_5series = float(1)/ref['5series']
#RF_7series = float(1)/ref['7series']
#RF_XJ = float(1)/ref['XJ']
#RF_Sclass = float(1)/ref['Sclass']
#
#
#
#PR_ES = 0.5 + 0.5*((PR_LS/RF_LS) + (PR_RX/RF_RX) + (PR_A8/RF_A8)
#
#
#
#
#
#
#ES = RF_ES
#LS = RF_LS
#RX = RF_RX
#A8 = RF_A8
#A6 = RF_A6
#3series = RF_3series
#5series = RF_5series
#7series = RF_7series
#XJ = RF_XJ
#Sclass = RF_Sclass