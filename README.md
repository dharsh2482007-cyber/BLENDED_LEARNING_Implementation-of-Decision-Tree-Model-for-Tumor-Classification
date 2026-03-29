# BLENDED_LEARNING
# Implementation of Decision Tree Model for Tumor Classification

## AIM:
To implement and evaluate a Decision Tree model to classify tumors as benign or malignant using a dataset of lab test results.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import required libraries and load the tumor dataset, then separate features (X) and target variable (y).


2.Split the dataset into training and testing sets.


3.Train the Decision Tree model using the training data.


4.Predict tumor classes and evaluate the model using accuracy and other performance metrics. 


## Program:
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVC
import seaborn as sns

data=pd.read_csv('food_items_binary.csv')
print(data.head())
print(data.columns)
features=['Calories','Total Fat', 'Saturated Fat', 'Sugars','Dietary Fiber','Protein' ]
target='class'

X=data[features]
y=data[target]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

svm=SVC()
param_grid={
    'C':[0.1,1,10,100],
    'kernel':['linear','rbf'],
    'gamma':['scale','auto']}
grid_search=GridSearchCV(svm,param_grid,cv=5,scoring='accuracy')
grid_search.fit(X_train,y_train)
best_model=grid_search.best_estimator_

print("Name: PRIYADHARSHINI")
print("Register Number:212225220076")
print("Best Parameters:",grid_search.best_params_)

y_pred=best_model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print("Name: P.PRIYADHARSHINI")
print("Register Number:212225220076")
print("Accuracy:",accuracy)
print("Classification Report:\n",classification_report(y_test,y_pred))

conf_matrix=confusion_matrix(y_test,y_pred)
sns.heatmap(conf_matrix,annot=True,fmt="d",cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

## Output:
<img width="785" height="612" alt="image" src="https://github.com/user-attachments/assets/502274d0-40d7-4817-8354-f6fac25991dd" />



## Result:
Thus, the Decision Tree model was successfully implemented to classify tumors as benign or malignant, and the model’s performance was evaluated.
