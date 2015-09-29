# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 17:50:24 2015

Purpose: Homework 3 Text Analytics Part A
"""
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


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
for i, value in enumerate(names):
    labels[i] = value
    G.add_node(i) #Adding the nodes

G.add_weighted_edges_from([(1,2,0.5),(3,2,1.75)])

pos = nx.spring_layout(G)

G.add_node(1)
G.add_node(2)
G.add_node(3)
G.add_edge(1,2, weight = 5)

G.add_edge(2,1, weight = 5)

nx.draw_networkx_labels(G,pos, labels, font_size=16)
nx.draw_networkx_nodes(G,pos,node_color='skyblue')
plt.figure(num=None, figsize=(10,9))
plt.show()

