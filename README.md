# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

step 1 - Start program.

step 2 - import pandas module and import the required data set.

step 3 - Find the null values and count them.

step 4 - Count number of left values.

step 5 - From sklearn import LabelEncoder to convert string

step 6 - Values to numerical values.

step 7 - From sklearn.model_selection import train_test_split.

step 8 - Assign the train dataset and test dataset.

step 9 - From sklearn tree import DecisionTreeClassifier.

step 10 - Use criteria as entropy.

step 11 - From sklearn import metrics.

step 12 - Find the accuracy of our model and predict the require values.

step 13 - Stop program.

## Program:

/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn
Developed by: ANUVIND KRISHNA.K
RegisterNumber: 212223080004

import pandas as pd
data = pd.read_csv("Employee.csv")
data
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()
x= data[["satisfaction_level","last_evaluation","number_
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,te
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

*/

## Output:
DATA:

![image](https://github.com/user-attachments/assets/bffe23de-fcff-463f-9cac-eeb024171abe)

ACCURACY:

![image](https://github.com/user-attachments/assets/c22c7e30-2a2f-4d55-afa7-fb0ba5ebdcd0)

PREDICT:

![image](https://github.com/user-attachments/assets/50c9a321-a68a-4d4f-b094-6d7a68dec5a4)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
