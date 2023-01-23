#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing important libraries
import numpy as np
import  pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , recall_score,f1_score,precision_score,confusion_matrix , classification_report
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib.inline', '')


# # IMPORTING DATASETS

# In[2]:


train_data=pd.read_csv('dataset.csv')


# In[3]:


train_data


# In[4]:


train_data=train_data.rename(columns={'sales':'department'})


# In[5]:


train_data


# In[6]:


train_data.describe()


# In[7]:


train_data.isnull().sum()


# In[8]:


train_data.info()


# In[9]:


train_data.columns


# In[10]:


train_data.nunique()


# # SPLITTING THE DATA INTO TARGET AND FEATURES

# In[11]:


X=train_data[['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident',
       'promotion_last_5years', 'department', 'salary']]


# In[12]:


y=train_data['left']


# # TRAIN TEST SPLIT 

# In[13]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=42)


# In[14]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[15]:


# splitting the train and test data into categorical andnumerical data
X_train_cat=X_train.select_dtypes(include='object')
X_train_num=X_train.select_dtypes(include=['int32','int64','float32','float64'])
X_test_cat=X_test.select_dtypes(include='object')
X_test_num=X_test.select_dtypes(include=['int32','int64','float32','float64'])


# In[16]:


X_train_num.reset_index(drop=True,inplace=True)
X_test_num.reset_index(drop=True,inplace=True)
X_train_cat.reset_index(drop=True,inplace=True)
X_test_cat.reset_index(drop=True,inplace=True)
y_train.reset_index(drop=True,inplace=True)
y_test.reset_index(drop=True,inplace=True)


# # PREPARING TRAIN AND TEST DATA

# In[24]:



X_train_cat_enc=pd.get_dummies(X_train_cat)

X_test_cat_enc=pd.get_dummies(X_test_cat)


# In[25]:


# concat the traoin and test data
X_train_cat_enc_df=pd.DataFrame(X_train_cat_enc)
X_test_cat_enc_df=pd.DataFrame(X_test_cat_enc)


# In[26]:


X_train_final=pd.concat([X_train_cat_enc_df,X_train_num],axis=1)
X_test_final=pd.concat([X_test_cat_enc_df,X_test_num],axis=1)


# In[27]:


X_train_final


# In[28]:


X_test_final


# # MODEL BUILDING AND PREDICTION

# In[30]:


model= LogisticRegression(solver='liblinear')
model.fit(X_train_final,y_train)
y_pred=model.predict(X_test_final)


# # MODEL EVALUATION

# In[51]:


ACCURACY = accuracy_score(y_test,y_pred)
RECALL= recall_score(y_test,y_pred)
PRECISION = precision_score(y_test,y_pred)
F1_SCORE = f1_score(y_test,y_pred)
CONFUSION_MATRIX=confusion_matrix(y_test,y_pred)
CLASSIFICATION_REPORT=classification_report(y_test,y_pred)


# In[52]:


print(ACCURACY)
print(RECALL)
print(PRECISION)
print(F1_SCORE)
print(CONFUSION_MATRIX)
print(CLASSIFICATION_REPORT)


# # RANDOM FOREST CLASSIFIER

# In[56]:


from sklearn.ensemble import RandomForestClassifier
model_1=RandomForestClassifier()
model_1.fit(X_train_final,y_train)
y_pred_1=model_1.predict(X_test_final)


# In[57]:


ACCURACY_1= accuracy_score(y_test,y_pred_1)
RECALL_1= recall_score(y_test,y_pred_1)
PRECISION_1 = precision_score(y_test,y_pred_1)
F1_SCORE_1 = f1_score(y_test,y_pred_1)
CONFUSION_MATRIX_1=confusion_matrix(y_test,y_pred_1)
CLASSIFICATION_REPORT_1=classification_report(y_test,y_pred_1)


# In[58]:


print(ACCURACY_1)
print(RECALL_1)
print(PRECISION_1)
print(F1_SCORE_1)
print(CONFUSION_MATRIX_1)
print(CLASSIFICATION_REPORT_1)


# In[ ]:




