#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Feature Extraction with PCA
import pandas as pd 
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.naive_bayes import GaussianNB

# load data
train = pd.read_csv('C:/Users/17708/Documents/R/MLProject_train (1).csv')
valid = pd.read_csv('C:/Users/17708/Documents/R/MLProject_valid.csv')
test = pd.read_csv('C:/Users/17708/Documents/R/MLProject_test (1).csv')

# Cleaning data
train["Z2"] = pd.to_numeric(train["Z2"])
#print (train["Z2"])
valid["Z2"] = pd.to_numeric(valid["Z2"])
#print (valid["Z2"])
test["Z2"] = pd.to_numeric(test["Z2"])
#print (test["Z2"])


#Train
ML_train = train.dropna()
Train = ML_train.drop(['target1', 'target2'], axis = 1)
#print(Train)
target1 = ML_train["target1"]
#print(target1)
target2 = ML_train["target2"]
#print(target2)

#Valid
ML_valid = valid.dropna()
Valid = ML_valid.drop(['target1', 'target2'], axis = 1)
#print(Valid)
target1v = ML_valid["target1"]
#print(target1v)
target2v = ML_valid["target2"]

#Test
ML_test = test.dropna()


# In[14]:


# Preprocessing
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
X_train = sc.fit_transform(Train) 
X_valid = sc.transform(Valid) 


# In[15]:


# Applying PCA function on training 
# and testing set of X component 
from sklearn.decomposition import PCA 
  
pca = PCA(n_components = 50) 
  
X_train = pca.fit_transform(X_train) 
X_valid = pca.transform(X_valid) 
  
explained_variance = pca.explained_variance_ratio_ 


# In[16]:


#Naive Bayes for Target1
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, target1)

#Predictions for Target1 & Accuracy
predicted1 = gnb.predict(X_valid)
predictedprobs = gnb.predict_proba(X_valid)


import numpy as np
print(np.mean(predicted1 == target1v))


# In[18]:


#ROC/AUC Target1
import sklearn.metrics as metrics
preds = predictedprobs[:,1]
fpr, tpr, threshold = metrics.roc_curve(target1v, preds)
roc_auc = metrics.auc(fpr, tpr)
print(roc_auc)


# In[19]:


#Confusion Matrix for Target1
from sklearn.metrics import confusion_matrix

print(confusion_matrix(target1v, predicted1))


# In[17]:


#Naive Bayes for Target2
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, target2)

#Predictions for Target1 & Accuracy
predicted2 = gnb.predict(X_valid)
predictedprobs2 = gnb.predict_proba(X_valid)


import numpy as np
print(np.mean(predicted2 == target2v))


# In[20]:


#ROC/AUC Target2
import sklearn.metrics as metrics
preds2 = predictedprobs2[:,1]
fpr2, tpr2, threshold2 = metrics.roc_curve(target2v, preds2)
roc_auc2 = metrics.auc(fpr2, tpr2)
print(roc_auc2)


# In[21]:


#Confusion Matrix for Target2
from sklearn.metrics import confusion_matrix

print(confusion_matrix(target2v, predicted2))


# In[ ]:




