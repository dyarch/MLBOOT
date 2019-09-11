
# coding: utf-8

# In[1]:


# data analysis and wrangling
import pandas as pd
import numpy as np

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#settings
pd.options.display.max_columns = 250
pd.options.display.max_rows = 250
sns.set(style="whitegrid")

import warnings
warnings.simplefilter('ignore')


# # Save Model Function

# In[16]:


import pickle
from sklearn import tree
# save the classifier
def save_model(model, name):
    path = r'{0}.pkl'.format(name)
    with open(path, 'wb') as fid:
        pickle.dump(model, fid) 
    
    # We are done so far.


# # Calculate Accuracy Function

# In[20]:


from sklearn import metrics
from sklearn.metrics import accuracy_score

def calculate_accuracy_confusion_matrix(model, test_features, test_labels):
    # predicting for test values
    DT_test_pred = model.predict(test_features)
    clf_DT_accuracy = accuracy_score(test_labels, DT_test_pred, normalize = True)
    print(clf_DT_accuracy)

    # Confusion Metrices for Decision Tree on train data.
    from sklearn import metrics
    from sklearn.metrics import confusion_matrix
    print("Confusion Metrices for Decision Tree")
    print("{0}".format(metrics.confusion_matrix(test_labels, DT_test_pred, labels=[1, 2, 3])))

    print("Classification Report")
    print("{0}".format(metrics.classification_report(test_labels, DT_test_pred, labels=[1, 2, 3])))


# In[4]:


df = pd.read_pickle("C:\Git Repos\OPIMLBoot\selected_feature.pkl")
df.head()


# In[5]:


df2 = df.copy()

# target value
Y = df2.UHGPrimacy.values
# features
X = df2.drop('UHGPrimacy', axis = 1).values


# In[6]:


from sklearn.model_selection import train_test_split
#Splitting the data.
split_test_size = 0.30

# Split the data into a training set and a test set
train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size =  split_test_size, random_state = 42)

# Check whether the correct % data split (70/30 %)Train vs Test data.
print("{0:0.2f}% in training".format(  len(train_features)/len(df2.index) *100))
print("{0:0.2f}% in testing".format(  len(test_features)/len(df2.index) *100))


# ## Logistic Regression

# In[21]:


# Logistic Regression algo
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score

#train the model.
model = LogisticRegression(C=0.01, solver='liblinear')  #.fit(X_train,y_train)
model.fit(train_features, train_labels.ravel())


# #### Performance on test set

# In[22]:


calculate_accuracy_confusion_matrix(model, test_features, test_labels)


# #### Save the model

# In[23]:


save_model(model, 'LR-70-30')


# # Decision Tree 

# In[24]:


# Decision Tree algo
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score

#train the model.
model = DecisionTreeClassifier(max_depth=7, random_state = 42, criterion='gini') # entropy
model.fit(train_features, train_labels)


# #### Performance on test set

# In[25]:


calculate_accuracy_confusion_matrix(model, test_features, test_labels)


# ### Save

# In[26]:


save_model(model, "DT-70-30")


# # Random Forest

# In[39]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(max_depth=8, n_estimators=25,min_samples_split=10, random_state=0)

model.fit(train_features, train_labels)


# #### Performance on test set

# In[40]:


calculate_accuracy_confusion_matrix(model, test_features, test_labels)


# #### Save the model

# In[41]:


save_model(model, 'RF-70-30')


# # Light GBM

# In[31]:


from lightgbm import LGBMClassifier

model=LGBMClassifier(n_estimators=300, learning_rate=0.01, num_leaves=80, max_depth = 7, colsample_bytree=0.2,
            reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)

model.fit(train_features, train_labels.ravel())


# #### Performance on test set

# In[32]:


calculate_accuracy_confusion_matrix(model, test_features, test_labels)


# #### Save

# In[33]:


save_model(model, 'LGBM-70-30')


# # XGBoost

# In[36]:


import xgboost as xgb

model = xgb.XGBClassifier(booster="gbtree", max_depth=7, objective="binary:logistic", random_state=42, nthread=4, n_estimators = 20, eta = 0.05)

model.fit(train_features, train_labels.ravel())


# ### Validation Accuracy

# In[37]:


calculate_accuracy_confusion_matrix(model, test_features, test_labels)


# ### Save Model

# In[38]:


save_model(model, 'XG-70-30')

