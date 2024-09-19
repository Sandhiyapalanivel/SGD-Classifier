# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step-1. Import Necessary Libraries and Load Data

Step-2. Split Dataset into Training and Testing Sets

Step-3. Train the Model Using Stochastic Gradient Descent (SGD)

Step-4. Make Predictions and Evaluate Accuracy

Step-5. Generate Confusion Matrix
## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: SANDHIYA P
RegisterNumber: 212223230183
*/
import pandas as pd 
from sklearn.datasets import load_iris 
from sklearn.linear_model import SGDClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, confusion_matrix 
import matplotlib.pyplot as plt
import seaborn as sns
iris=load_iris()

#Create a Pandas DataFrame

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train, y_train)
y_pred = sgd_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
```

## Output:
![image](https://github.com/user-attachments/assets/61db8c03-fcfe-4616-b598-76616c935042)

```
Accuracy
```
![image](https://github.com/user-attachments/assets/ab98bc77-f186-40e7-9445-231c8bf15514)

```
Confusion matix
```
![image](https://github.com/user-attachments/assets/c0ab00cf-d9ba-476f-8a1a-ac3c045ac2d7)



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
