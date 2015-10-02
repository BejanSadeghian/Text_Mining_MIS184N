# -*- coding: utf-8 -*-
"""
Created on Thu Oct 01 19:21:17 2015

Purpose: Homework 3 Part C
"""

import pandas as pd


rawdata = pd.read_csv(r'file:///C:\Users\beins_000\Documents\GitHub\Text_Mining_MIS184N\Group_Assignment_3\Assignment%203%20Edmunds%20Posts.csv')

rawdata = rawdata.ix[:,0]

N_before = 3
N_after = 3

models = ['es', 'ls', 'rx', 'a8', 'a6', '3series', '5series', '7series', 'xj', 'sclass']
make = ['audi', 'bmw', 'jaguar', 'lexus', 'mercedes']

rawdata.ix[0,0].split()
for row in rawdata:
    words = row.split()
    for word in words:
        if word in 