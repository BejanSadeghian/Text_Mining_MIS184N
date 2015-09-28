
# coding: utf-8

# In[83]:

import pandas as pd
import numpy as np
from pandas import Series
from pandas import DataFrame
from patsy import dmatrices
from sklearn.cross_validation import train_test_split,StratifiedShuffleSplit,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import neighbors
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
#from sklearn.cross_validation import 
#from sklearn.cross_validation import 
import scipy.sparse
import sklearn.cluster
from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
import itertools
import math


# In[2]:

yelp = pd.read_csv("yelp.csv")


# In[3]:

high_mask = yelp['stars'] > 3
yelp['High'] = 0
yelp.ix[high_mask, 'High'] = 1


# #Task A

# In[4]:

formula = 'High ~ 0 + votes_cool + votes_funny + votes_useful + Cheap + Moderate + Expensive  ' + ' + VeryExpensive + American + Chinese + French + Japanese + Indian + Italian + Greek ' + ' + Mediterranean + Mexican + Thai + Vietnamese + Others'


# In[5]:

Y, X = dmatrices(formula, yelp, return_type='dataframe')
y = Y['High'].values


# In[6]:

index = StratifiedShuffleSplit(y, n_iter = 1, test_size = 0.3, train_size = 0.7)


# In[7]:

index


# In[8]:

DF_X = DataFrame(X)


# In[9]:

for train_index, test_index in index:
    print(train_index)
    X_train, X_test = DF_X.iloc[train_index,], DF_X.iloc[test_index,]
    y_train, y_test = y[train_index], y[test_index]


# In[10]:

X_train.head()


# ###Logistic Regression

# In[11]:

logistic_model = LogisticRegression()
logistic_result = logistic_model.fit(X_train, y_train)


# In[12]:

logistic_train_prediction = logistic_model.predict(X_train)
print metrics.accuracy_score(y_train, logistic_train_prediction)


# In[13]:

logistic_test_prediction = logistic_model.predict(X_test)
print metrics.accuracy_score(y_test, logistic_test_prediction)


# ###KNN

# In[14]:

knn_model = neighbors.KNeighborsClassifier(n_neighbors=15, weights='uniform', p=2)
knn_result = knn_model.fit(X_train, y_train)


# In[15]:

knn_train_prediction = knn_model.predict(X_train)
print metrics.accuracy_score(y_train, knn_train_prediction)


# In[16]:

knn_test_prediction = knn_model.predict(X_test)
print metrics.accuracy_score(y_test, knn_test_prediction)


# #Task B: Classification on Text

# ###Random Sampling

# In[17]:

np.random.seed(1234567)
train = yelp.sample(int(len(yelp)*0.7), replace=False)


# In[18]:

test = yelp[~yelp.index.isin(train.index.values)]


# In[19]:

train_x = train['Review']
train_y = train['High']
test_x = test['Review']
test_y = test['High']


# In[20]:

vectorizer = TfidfVectorizer(min_df=0,smooth_idf=True, strip_accents='unicode', norm='l2')


# In[21]:

def text_classification (v):
    X_transform=v.fit_transform(train_x)
    X_test=v.transform(test_x)
    
    nb_classifier = MultinomialNB().fit(X_transform, train_y)
    y_nb_predicted = nb_classifier.predict(X_test)
    
    predict_y=Series(y_nb_predicted).reset_index()[0]
    df=pd.DataFrame()
    df['Predicted']=predict_y
    df['Actual']=test_y.reset_index()['High']
    
    print "Percent Correct\n",round((df['Predicted']==df['Actual']).mean()*100,3)
    print "\nConfusion Matrix\n",pd.crosstab(index=df['Actual'],columns=df['Predicted'])
    print "\nProportion Table\n", pd.crosstab(index=df['Actual'],columns=df['Predicted']).apply(lambda r: r/r.sum(), axis=1)


# In[22]:

text_classification(vectorizer)


# ###Undersampling data set to ensure 50/50 split

# In[23]:

highs=yelp[yelp['High']==1]
lows=yelp[yelp['High']==0]


# In[24]:

sample_high=highs.sample(len(lows),replace=False).copy()


# In[25]:

sample=sample_high.append(lows, ignore_index=True)


# In[26]:

train=sample.sample(int(0.7*len(sample)),replace=False).copy()
test=sample[~sample.index.isin(train.index.values)].copy()


# In[27]:

train_x = train['Review']
train_y = train['High']
test_x = test['Review']
test_y = test['High']


# In[28]:

text_classification(vectorizer)


# #Task C

# In[29]:

def yelper(removeList,sample=sample):
    train=sample.sample(int(0.7*len(sample)),replace=False).copy()
    test=sample[~sample.index.isin(train.index.values)].copy()
    
    train_x = train['Review']
    train_y = train['High']
    test_x = test['Review']
    test_y = test['High']
    
    X_transform=vectorizer.fit_transform(train_x) #just the reviews
    sam=train.drop(removeList, axis=1).copy()
    samsparse=scipy.sparse.csr_matrix(sam.to_sparse()) #sparsing the numeric
    surprise=scipy.sparse.hstack([X_transform, samsparse]) #combining numeric and text
    
    X_test=vectorizer.transform(test_x) #repeating above for test
    samt=test.drop(removeList, axis=1).copy() #removing irrelevant
    samsparset=scipy.sparse.csr_matrix(samt.to_sparse()) #sparsing numeric 
    surpriset=scipy.sparse.hstack([X_test, samsparset]) #combining numeric and text
    
    nb_classifier = MultinomialNB().fit(surprise, train_y)
    y_nb_predicted = nb_classifier.predict(surpriset)
    
    predict_y=Series(y_nb_predicted).reset_index()[0]
    df=pd.DataFrame()
    df['Predicted']=predict_y
    df['Actual']=test_y.reset_index()['High']
    
    print "Percent Correct\n",round((df['Predicted']==df['Actual']).mean()*100,3)
    print "\nConfusion Matrix\n",pd.crosstab(index=df['Actual'],columns=df['Predicted'])
    print "\nProportion Table\n", pd.crosstab(index=df['Actual'],columns=df['Predicted']).apply(lambda r: r/r.sum(), axis=1)


# In[30]:

rL=["Review","stars","High"]
yelper(rL)


# In[31]:

skf = StratifiedKFold(sample['High'], n_folds=3, random_state=45)


# ##Part D

# In[32]:

rawsentiment = pd.read_excel('Yelp_Review_Data_Results.xlsx')


# In[33]:

rawsentiment['True_Sentiment'] = rawsentiment['Pos Senti'] + rawsentiment['Neg Senti']


# In[34]:

rawsentiment['Actual'] = yelp['High']


# ##Raw Sentiment, 0 defaults to low

# In[35]:

rawsentiment['Predicted']=0
rawsentiment['Predicted'].ix[rawsentiment['True_Sentiment']>0]=1
df=rawsentiment
print "Percent Correct\n",round((df['Predicted']==df['Actual']).mean()*100,3)
print "\nConfusion Matrix\n",pd.crosstab(index=df['Actual'],columns=df['Predicted'])
print "\nProportion Table\n", pd.crosstab(index=df['Actual'],columns=df['Predicted']).apply(lambda r: r/r.sum(), axis=1)


# #Raw Sentiment, 0 defaults to high

# In[36]:

rawsentiment['Predicted']=0
rawsentiment['Predicted'].ix[rawsentiment['True_Sentiment']<0]=1
df=rawsentiment
print "Percent Correct\n",round((df['Predicted']==df['Actual']).mean()*100,3)
print "\nConfusion Matrix\n",pd.crosstab(index=df['Actual'],columns=df['Predicted'])
print "\nProportion Table\n", pd.crosstab(index=df['Actual'],columns=df['Predicted']).apply(lambda r: r/r.sum(), axis=1)


# ##Part E

# In[37]:

DTMReviews = vectorizer.fit_transform(sample['Review'])


# In[38]:

#from sklearn.metrics.pairwise import cosine_similarity
#def new_euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False):
#    return cosine_similarity(X,Y)

# monkey patch (ensure cosine dist function is used)
#from sklearn.cluster import k_means_k_means_.euclidean_distances
#k_means_.euclidean_distances = new_euclidean_distances 


# In[39]:

Cluster = sklearn.cluster.KMeans(n_clusters=2, random_state = 1)
clusterout = Cluster.fit(DTMReviews)


# In[40]:

series_clusters = Series(clusterout.labels_)


# In[41]:

sample['Cluster'] = series_clusters


# In[42]:

test = pd.concat([sample, series_clusters], axis=1)


# In[43]:

sample.head()


# In[45]:

print "Confusion Matrix \n", pd.crosstab(index=sample['High'],columns=sample['Cluster'])
print "\nPercent Correct (if cluster 1 = low)\n", round((sample['High']!=sample['Cluster']).mean()*100,3)
print "\nPercent Correct (if cluster 1 = high)\n", round((sample['High']==sample['Cluster']).mean()*100,3)
print "\nProportion Table\n", pd.crosstab(index=sample['High'],columns=sample['Cluster']).apply(lambda r: r/r.sum(), axis=1)


# ##Part F

# In[46]:

#from nltk import download
#from textblob_aptagger import PerceptronTagger
#from textblob import Blobber


# In[147]:

reviews=yelp['Review']
reviews=reviews.str.decode("utf-8")
reviews_high = reviews[yelp['High'] == 1].copy()
reviews_low = reviews[yelp['High'] == 0].copy()
reviews=list(reviews)


# In[148]:

token_high = reviews_high.map(word_tokenize)
token_low = reviews_low.map(word_tokenize)
token_high = list(token_high)
token_low = list(token_low)


# In[149]:

flat_high = list(itertools.chain.from_iterable(token_high))
flat_low = list(itertools.chain.from_iterable(token_low))
high_lower = [t.lower() for t in flat_high if t.isalpha()]
low_lower = [t.lower() for t in flat_low if t.isalpha()]


# In[150]:

high_vc = Series(high_lower).value_counts()
low_vc = Series(low_lower).value_counts()


# In[151]:

high_clean = [word for word in high_vc.index if word not in stopwords.words('english')]
low_clean = [word for word in low_vc.index if word not in stopwords.words('english')]


# In[152]:

tag_high100 = pos_tag(high_clean[:100])
tag_low100 = pos_tag(low_clean[:100])


# In[153]:

high_noun = [word for word,tag in tag_high100 if tag == 'NN']
low_noun = [word for word,tag in tag_low100 if tag == 'NN']


# In[156]:

print high_noun[:30] #Food, Place/Atmosphere, Service, Menu, Wait Time


# In[157]:

print low_noun[:30] #Food, Place/Atmosphere, Service, Order, Cleanliness


# In[ ]:

def noun_extractor(series):
    token = series.map(word_tokenize)
    tag = token.map(pos_tag)
    [x for x in tag]


# In[121]:

number_of_hw = sum(high_vc)
number_of_lw = sum(low_vc)
mask = (high_vc > 10) | (low_vc > 10)

high_vc_norm = (1+high_vc)/number_of_hw
low_vc_norm = (1+low_vc)/number_of_lw


# ##Task F (extra): Finding Top Tokens Predictive of High/Low Ratings

# In[136]:

concat_vc = pd.concat([high_vc,low_vc],join='outer', axis=1).fillna(0)
concat_vc.rename(columns={0:'High',1:'Low'}, inplace=True)
mask = (concat_vc['High'] > 10) | (concat_vc['Low'] > 10)
concat_vc = concat_vc[mask]

high_vc_norm = (1+concat_vc['High'])/number_of_hw
low_vc_norm = (1+concat_vc['Low'])/number_of_lw


# In[137]:

#Concat two series
concat_high_low = pd.concat([high_vc_norm,low_vc_norm],join='outer', axis=1)
concat_high_low.rename(columns={0:'High',1:'Low'}, inplace=True)
concat_high_low['High'].fillna(0.000001, inplace=True)
concat_high_low['Low'].fillna(0.000001, inplace=True)
concat_high_low['HL_Ratio']= (concat_high_low['High'].map(math.log)-concat_high_low['Low'].map(math.log))


# In[138]:

concat_high_low.sort('HL_Ratio', inplace=True)


# In[139]:

low_100ratio = concat_high_low[:100]
high_100ratio = concat_high_low[-100:]
low_clean_2 = [word for word in low_100ratio.index if word not in stopwords.words('english')]
high_clean_2 = [word for word in high_100ratio.index if word not in stopwords.words('english')]


# In[140]:

tag_high100_2 = pos_tag(high_clean_2[:100])
tag_low100_2 = pos_tag(low_clean_2[:100])
high_noun_2 = [word for word,tag in tag_high100_2 if tag == 'NN']
low_noun_2 = [word for word,tag in tag_low100_2 if tag == 'NN']


# In[145]:

low_noun_2[:10]


# In[146]:

high_noun_2[:10]


# In[ ]:



