# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 23:02:52 2021

@author: Manju
"""

import pandas as pd
import seaborn as sns

HR_data = pd.read_csv("D:/data/HR_Data.csv")
HR_data.head(5)
HR_data.tail(5)
# plots 
sns.countplot(x="left",data=HR_data)
sns.countplot(x="left",hue="salary",data=HR_data)
sns.countplot(x="left",hue="exp_in_company",data= HR_data)
sns.countplot(x="left",hue="promotion_last_5years",data=HR_data)
sns.countplot(x="left",hue="role",data=HR_data)
#histogram
HR_data.info()
HR_data["average_montly_hours"].plot.hist()
#to check missing values
HR_data.isnull()
HR_data.isnull().sum()

pd.get_dummies(HR_data["role"])
pd.get_dummies(HR_data["role"],drop_first=True)
Role_Dummy = pd.get_dummies(HR_data["role"],drop_first=True)
Role_Dummy.head(5)

pd.get_dummies(HR_data["salary"])
pd.get_dummies(HR_data["salary"],drop_first=True)
Salary_Dummy = pd.get_dummies(HR_data["salary"],drop_first=True)
Salary_Dummy.head(5)

HR_data = pd.concat([HR_data,Role_Dummy,Salary_Dummy],axis=1)
HR_data.head(5)

HR_data.drop(["role","salary"],axis=1,inplace=True)
HR_data.head(5)

x=HR_data.drop("left",axis=1)
y=HR_data["left"]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
#Hence, accuracy = (2651+343)/(2651+230+526+343)=79.84%

print(logmodel.coef_)
print(logmodel.intercept_)

# backward elimination
HR_data_1 = HR_data
HR_data_1.head(5)

import statsmodels.api as sm

x1=HR_data_1.drop("left",axis=1)
y1=HR_data_1["left"]
import numpy as nm
x1 = nm.append(arr = nm.ones((14999,1)).astype(int), values=x1, axis=1)

x_opt= x1[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]

regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()

regressor_OLS.summary()

x_opt= x1[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()


x_opt= x1[:, [0,1,2,3,4,5,6,7,8,9,10,11,14,15,16,17,18]]
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()


x_opt= x1[:, [0,1,2,3,4,5,6,7,8,10,11,14,15,16,17,18]]
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()


x_opt= x1[:, [0,1,2,3,4,5,6,7,8,10,11,16,17,18]]
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()


x_opt= x1[:, [0,1,2,3,4,5,6,7,8,10,11,17,18]]
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

#Hence,independent var - satisfactory_level,last_evaluation,number_projects,avg-monthly-hours,exp-in-company,work-accident,promotion in last 5years are significant variable 
#for the predicting the value of Dependent Var "left".
#So we can now predict efficiently using these variables.

from sklearn.model_selection import train_test_split
x_BE_train, x_BE_test, y_BE_train, y_BE_test= train_test_split(x_opt, y1, test_size= 0.25, random_state=0)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_BE_train, y_BE_train)

predictions = logmodel.predict(x_BE_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_BE_test,predictions)

#Accuracy = (2656+315)/(2656+225+554+315) = 79.22%
print(logmodel.coef_)
print(logmodel.intercept_)


#So, ur final Predicitve Modelling Equation becomes:
    #left =
 # exp^(-0.66-4.15*satisfactory level+0.72*last evaluation-0.33*project numbers+0.004*avg monthly hrs+0.27*exp in company-1.44*work accident-1.04promotion last 5 years-0.65R&D+0.17*HR+-0.44*management+1.80*low+1.26*medium)
 #/exp^(-0.66-4.15*satisfactory level+0.72*last evaluation-0.33*project numbers+0.004*avg monthly hrs+0.27*exp in company-1.44*work accident-1.04promotion last 5 years-0.65R&D+0.17*HR+-0.44*management+1.80*low+1.26*medium)+1