#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install catboost
# !pip install category_encoders
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import category_encoders as ce
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, f1_score,jaccard_score,multilabel_confusion_matrix,log_loss
import lightgbm as lgbm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv('/users/pgrad/mohanni/demo/thesis/data/adult.data',header=None)
df.head(3)


# In[9]:


#encodings and split

def onehot_all(X,y,ratio):
    #split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=42)
    
    obj_cols=X.select_dtypes('object').columns
    enc=ce.OneHotEncoder(cols=obj_cols,handle_missing='return_nan').fit(X_train,y_train)
    X_train=enc.transform(X_train)
    X_test=enc.transform(X_test)
    
    return X_train, X_test,y_train, y_test

def target_all(X,y,ratio):
    #split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=42)
    
#     obj_cols=X.select_dtypes('object').columns
    enc=ce.TargetEncoder(handle_missing='return_nan').fit_transform(X_train,y_train)
#     X_train=enc.transform(X_train)
    X_test=enc.transform(X_test)
    
    return X_train,X_test,y_train, y_test
    
def onehot_target(X,y,ratio,thresh):
    #split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=42)
    
    low_card_cols,high_card_cols=[],[]
    obj_cols=X.select_dtypes('object').columns
    for col in obj_cols:
        if X_train[col].nunique()<=thresh:
            low_card_cols.append(col)
        else:
            high_card_cols.append(col)
    
    
    enc=ce.OneHotEncoder(cols=low_card_cols,handle_missing='return_nan').fit_transform(X_train,y_train)
    X_test=enc.transform(X_test)
    
    enc=ce.TargetEncoder(cols=high_card_cols,handle_missing='return_nan').fit_transform(X_train,y_train)
    X_test=enc.transform(X_test)
            


# In[5]:



def train_evaluate( X_train,X_test,y_train, y_test):
    print("catboost training with gridsearch")

    model = CatBoostClassifier(task_type="GPU", devices='0:1')
    grid = GridSearchCV(estimator=model, param_grid = parameters)
    grid.fit(X_train, y_train,verbose=0)    

    model=grid.best_estimator_
    evaluate(model,X_test,y_test)

    print("lightgbm training with gridsearch")

    model = lgbm.LGBMClassifier(boosting_type='goss')
    grid = GridSearchCV(estimator=model, param_grid = parameters)
    grid.fit(X_train, y_train,verbose=0)    

    model=grid.best_estimator_
    evaluate(model,X_test,y_test)

    print("xgboost training with gridsearch")

    model = XGBClassifier(tree_method='gpu_hist', gpu_id=0)
    grid = GridSearchCV(estimator=model, param_grid = parameters)
    grid.fit(X_train, y_train,verbose=0)    

    model=grid.best_estimator_
    evaluate(model,X_test,y_test)


# In[6]:


#evaluation function

def evaluate(model,X_test,y_test):
    
    pred=model.predict(X_test)
    pred_proba=model.predict_proba(X_test)

    print('accuracy:',accuracy_score(y_test,pred))
    print('f1 macro:',f1_score(y_test,pred, average='macro'))
    print('f1_micro:',f1_score(y_test,pred, average='micro'))
    print('log_loss:',log_loss(y_test,pred_proba))


# In[7]:


# print("catboost training")
# model_cat = CatBoostClassifier(iterations=10,max_depth=10, task_type="GPU", devices='0:1')
# model_cat.fit(X_train, y_train, verbose=0)

#evaluate
# evaluate(model_cat,X_test,y_test)

# print("lightgbm training")

# model_lgbm = lgbm.LGBMClassifier(boosting_type='goss',max_depth=10,n_estimators=10)
# model_lgbm.fit(X_train, y_train)

#evaluate
# evaluate(model_lgbm,X_test,y_test)

# print("xgboost training")

# model_xgb = XGBClassifier(max_depth=10,n_estimators=10,tree_method='gpu_hist', gpu_id=0)
# model_xgb.fit(X_train, y_train)

#evaluate
# evaluate(model_xgb,X_test,y_test)


# In[8]:


X=df.iloc[:,:-1]
y=df.iloc[:,-1:]
# X_train,X_test,y_train, y_test=onehot_all(df.iloc[:,:-1],y,.2)
# X_train,X_test,y_train, y_test=target_all(df.iloc[:,:-1],y,ratio=.2)
# X_train,X_test,y_train, y_test=onehot_target(df.iloc[:,:-1],y,ratio=.2,thresh=10)


parameters = {'depth'         : [6,8,10,12],
              'learning_rate' : [.01,.05,.1,.2],
              'iterations'    : [100,500,1000]
              }

print("first with one hot for all")
X_train,X_test,y_train, y_test=onehot_all(X,y,.2)
train_evaluate( X_train,X_test,y_train, y_test)

# print("now with target encoding")
# X_train,X_test,y_train, y_test=target_all(X,y,ratio=.2)
# train_evaluate( X_train,X_test,y_train, y_test)


# In[ ]:




