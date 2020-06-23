# -*- coding: utf-8 -*-
"""traffic_violations_lgbm.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uxNWm-OZaXLNIVP7eMtYEorcAQBJDvJZ
"""

#!pip install catboost
#!pip install category_encoders
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from category_encoders import *
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, f1_score,jaccard_score,multilabel_confusion_matrix
import lightgbm as lgbm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np


df=pd.read_csv('https://data.montgomerycountymd.gov/api/views/4mse-ku6q/rows.csv?accessType=DOWNLOAD')
# df.to_csv('data')
# df=pd.read_csv('data')

#solving problem of unseen classes in label encoding
#https://stackoverflow.com/questions/21057621/sklearn-labelencoder-with-never-seen-before-values

df1=df.loc[:,[ 'Description','Belts','Property Damage','Fatal','Commercial License','HAZMAT','Commercial Vehicle','Alcohol','Work Zone','Year','Race','Gender','Arrest Type','Violation Type']].dropna()

X=df1.iloc[:,:-1]
y=df1.iloc[:,-1]


for col in ['Belts','Property Damage','Fatal','Commercial License','HAZMAT','Commercial Vehicle','Alcohol','Work Zone']:
  X[col]=X[col].map(lambda x: 1 if x=='Yes' else 0)
X['Gender']=X['Gender'].map(lambda x: 1 if x=='M' else 2 if x=='F' else 3)

X=pd.get_dummies(X,columns=['Race'])

#split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02, random_state=42)

encoder=TargetEncoder()
encoder.fit(X_train,y_train,cols=['Arrest Type','Violation Type'])
X_train=encoder.transform(X_train)
X_test=encoder.transform(X_test)

#xgb

model_xgb = XGBClassifier(tree_method='gpu_hist', gpu_id=0)
parameters = {'depth'         : [3,8],
              'learning_rate' : [0.1],
              'iterations'    : [100]
              }
grid = GridSearchCV(estimator=model_xgb, param_grid = parameters)
grid.fit(X_train, y_train)    

# Results from Grid Search
print("\n========================================================")
print(" Results from Grid Search " )
print("========================================================")    

print("\n The best estimator across ALL searched params:\n",
      grid.best_estimator_)

print("\n The best score across ALL searched params:\n",
      grid.best_score_)

print("\n The best parameters across ALL searched params:\n",
      grid.best_params_)

print("\n ========================================================")

c=grid.best_estimator_
print("training")
print("macro f1: ",f1_score(y_train,c.predict(X_train), average='macro'))
print("micro f1: ",f1_score(y_train, c.predict(X_train), average='micro'))
print("macro jaccard: ",jaccard_score(y_train,c.predict(X_train), average='macro'))
print("micro jaccard: ",jaccard_score(y_train, c.predict(X_train), average='micro'))

print("test")
print("macro f1: ",f1_score(y_train,c.predict(X_train), average='macro'))
print("micro f1: ",f1_score(y_train, c.predict(X_train), average='micro'))
print("macro jaccard: ",jaccard_score(y_train,c.predict(X_train), average='macro'))
print("micro jaccard: ",jaccard_score(y_train, c.predict(X_train), average='micro'))

print(multilabel_confusion_matrix(y_test, c.predict(X_test)))