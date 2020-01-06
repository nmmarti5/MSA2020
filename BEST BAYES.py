#!/usr/bin/env python
# coding: utf-8

# In[21]:


#Importing Libraries & Data 
import pandas as pd
train = pd.read_csv('C:/Users/17708/Documents/R/MLProject_train (1).csv')
valid = pd.read_csv('C:/Users/17708/Documents/R/MLProject_valid.csv')
test = pd.read_csv('C:/Users/17708/Documents/R/MLProject_test (1).csv')


# In[22]:


#Make Z2 Numerical instead of a string
train["Z2"] = pd.to_numeric(train["Z2"])
#print (train["Z2"])
valid["Z2"] = pd.to_numeric(valid["Z2"])
#print (valid["Z2"])
test["Z2"] = pd.to_numeric(test["Z2"])
#print (test["Z2"])


# In[23]:


#Dropping N/As & Seperating Targets (train)

ML_train = train.dropna()
Train = ML_train.drop(['target1', 'target2'], axis = 1)
#print(Train)
target1 = ML_train["target1"]
#print(target1)
target2 = ML_train["target2"]
#print(target2)


# In[24]:


#Dropping N/As & Seperating Targets (valid & test)

#Valid
ML_valid = valid.dropna()
Valid = ML_valid.drop(['target1', 'target2'], axis = 1)
#print(Valid)
target1v = ML_valid["target1"]
#print(target1v)
target2v = ML_valid["target2"]
#print(target2v)

#Test
Test = test.dropna()


# In[25]:


#Naive Bayes for Target1
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(Train, target1)

#Predictions for Target1 & Accuracy
predicted1 = gnb.predict(Valid)
predictedprobs = gnb.predict_proba(Valid)
predictedprobs1test = gnb.predict_proba(Test)

import numpy as np
print(np.mean(predicted1 == target1v))


# In[26]:


#ROC/AUC Target1
import sklearn.metrics as metrics
preds = predictedprobs[:,1]
fpr, tpr, threshold = metrics.roc_curve(target1v, preds)
roc_auc = metrics.auc(fpr, tpr)
print(roc_auc)


# In[27]:


#Confusion Matrix for Target1
from sklearn.metrics import confusion_matrix

print(confusion_matrix(target1v, predicted1))


# In[28]:


#Naive Bayes for Target2
from sklearn.naive_bayes import GaussianNB
gnb2 = GaussianNB()
gnb2.fit(Train, target2)

#Predictions for Target2 & Accuracy 
import numpy as np

predicted2 =gnb2.predict(Valid)
predictedprobs2 = gnb2.predict_proba(Valid)
predictedprobs2test = gnb2.predict_proba(Test)

print(np.mean(predicted2 == target2v))


# In[29]:


#ROC/AUC Target2
import sklearn.metrics as metrics
preds2 = predictedprobs2[:,1]
fpr2, tpr2, threshold2 = metrics.roc_curve(target2v, preds2)
roc_auc2 = metrics.auc(fpr2, tpr2)
print(roc_auc2)


# In[30]:


#Confusion Matrix for Target2
from sklearn.metrics import confusion_matrix

print(confusion_matrix(target2v, predicted2))


# In[ ]:




