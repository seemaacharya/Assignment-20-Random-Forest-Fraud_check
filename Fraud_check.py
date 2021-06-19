# -*- coding: utf-8 -*-
"""
Created on Wed May 26 15:28:09 2021

@author: DELL
"""

#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#loading the dataset
df=pd.read_csv("Fraud_check.csv")
df.head()

#Creating dummy variable for Undergrad,Marital.Status,Urban and dropping the first dummy variable
df=pd.get_dummies(df,columns=['Undergrad','Marital.Status','Urban'],drop_first=True)

#Creating new columns TaxInc and dividing TaxableIncome on the basis of [10002,30000,99620] for Risky and Good
df['TaxInc']=pd.cut(df['Taxable.Income'],bins=[10002,30000,99620],labels=['Risky','Good'])

#lets assume:taxable_income<=30000 as Risky=0 and others are Good=1
#After creation of new column TaxInc also made its dummies var concating right side of df
df = pd.get_dummies(df,columns=['TaxInc'],drop_first=True)
df.tail(10)

#Normalization function
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)

#Normalize the dataframe(considering only the numeric data)
df_norm=norm_func(df.iloc[:,1:])
df_norm.tail(10)

#Declaring the x and y
x= df_norm.drop(['TaxInc_Good'], axis=1)
y=df_norm['TaxInc_Good']

#train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=0)

#Model building using RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
num_trees=100
max_features=3
kfold = KFold(n_splits=10, random_state=0)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
model.fit(X_train,y_train)
pred=model.predict(X_test)

#Evaluating the model using cross_val_score
results = cross_val_score(X_train,y_train)
print(results.mean())
#72.83%






