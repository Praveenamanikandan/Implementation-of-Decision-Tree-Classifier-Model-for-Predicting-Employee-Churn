# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 1.import pandas module and import the required data set.

2.Find the null values and count them.

3.Count number of left values.

4.From sklearn import LabelEncoder to convert string values to numerical values.

5.From sklearn.model_selection import train_test_split.

6.Assign the train dataset and test dataset.

7.From sklearn.tree import DecisionTreeClassifier.

8.Use criteria as entropy.

9.From sklearn import metrics.

10.Find the accuracy of our model and predict the require values.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Praveena M
RegisterNumber:  2122230400153
*/
```
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()
## Output:
![image](https://github.com/user-attachments/assets/fc450323-94fe-4e46-9f4e-34bdcdfb76e2)

```
data.info()
```
## Output:
![image](https://github.com/user-attachments/assets/ce2e2f14-7f5f-4387-a782-7083f5efb3ea)

```
data.isnull().sum()
```
## Output:
![image](https://github.com/user-attachments/assets/7a09659c-b71a-405f-a108-521a4de13012)
```
data["left"].value_counts()
```
## Output:
![image](https://github.com/user-attachments/assets/00c18aeb-6f68-4255-8f29-807f9fc9ebac)
```
from sklearn.preprocessing import LabelEncoder 
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
```
# Output:
![image](https://github.com/user-attachments/assets/e49f7a28-cdb1-49f7-bbda-bbf900833b65)
```
x=data[["satisfaction_level", "last_evaluation", "number_project","average_montly_hours", "time_spend_company", "Work_accident", "promotion_last_5years", "salary"]]

x.head()
```
## Output:
![image](https://github.com/user-attachments/assets/dcdf1bea-7852-41f1-9a94-c16907c1707c)
```
y=data["left"]
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=100)
from sklearn.tree import DecisionTreeClassifier 
dt=DecisionTreeClassifier (criterion="entropy") 
dt.fit(x_train, y_train) 
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```
## Output:
![image](https://github.com/user-attachments/assets/89ef23d0-1fc6-4f95-ad4a-48871aa529be)
```
dt.predict([[0.5,0.8,9,260,6,0,1,]])
```
## Output:
![image](https://github.com/user-attachments/assets/a437f4ec-6803-4acf-9657-a3df84028d2d)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
