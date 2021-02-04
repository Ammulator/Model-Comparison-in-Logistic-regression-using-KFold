#1. Importing Libraries
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score,precision_score,accuracy_score,f1_score

#2. importing dataset and using Kfold function

df=pd.read_csv("C:/Users/Yuvraj singh/Desktop/california_housing_train.csv")
kf=KFold(n_splits=5,shuffle=True)

#3. Selecting Features for Different Models

X1=df[['housing_median_age','total_rooms','total_bedrooms','population','households','median_income']].values
X2=df[['housing_median_age','total_rooms','total_bedrooms']].values
X3=df[['total_bedrooms','population']].values
y=df['median_house_value'].values

#4. Now we create a score model function for selecting the models

def score_model(X, y, kf):
  accuracy_scores = [] 
  precision_scores = []
  recall_scores = []
  f1_scores = [] 
  for train_index, test_index in kf.split(X): 
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = LogisticRegression() 
    model.fit(X_train, y_train) 
    y_pred = model.predict(X_test) 
    accuracy_scores.append(accuracy_score(y_test, y_pred)) 
    precision_scores.append(precision_score(y_test, y_pred,average='micro')) 
    recall_scores.append(recall_score(y_test, y_pred,average='micro')) 
    f1_scores.append(f1_score(y_test, y_pred,average='micro'))
    
  print("accuracy:", np.mean(accuracy_scores)) 
  print("precision:", np.mean(precision_scores)) 
  print("recall:", np.mean(recall_scores)) 
  print("f1 score:", np.mean(f1_scores))

#5. Calling Function

print("Logistic Regression for all features: ")
score_model(X1,y,kf)
print("Logistic Regression for housing_median_age, total_rooms, total_bedrooms are: ")
score_model(X2,y,kf)
print("Logistic Regression for total_bedrooms, population are :")
score_model(X3,y,kf)

#Logistic Regression for Best Model
model=LogisticRegression()
model.fit(X1,y)
model.predict(20,1387,236,1841,633,1.82)
