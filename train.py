import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
# import missingno as msno
import plotly.express as px
from plotly.offline import init_notebook_mode

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

## Pipeline
from sklearn.pipeline import Pipeline , FeatureUnion
from sklearn_features.transformers import DataFrameSelector

## preprocessing
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import RobustScaler,OrdinalEncoder #, OneHotEncoder, LabelEncoder

## imbalanced data
from imblearn.over_sampling import SMOTE

## metrics
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

## tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

import skops.io as sio

loan_df = pd.read_csv("data/loan_sanction_train.csv")
loan_df.drop("Loan_ID", axis=1, inplace=True)
loan_df = loan_df.sample(frac=1)
print(loan_df.head(3))

# Feature engineering
data = loan_df.copy()
data["TotalIncome"] = data["ApplicantIncome"] + data["CoapplicantIncome"]
# sns.kdeplot(data=data, x="TotalIncome", fill=True)
# plt.title("Total Income Distribution before log transform".title())
# plt.show()
data["EMI"] = data["LoanAmount"] / data["Loan_Amount_Term"]
print(data.columns.to_list())

# # Split data
X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"] ## Target
## split the data to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state=42, stratify=y)

print("X train shape : ", X_train.shape)
print("y train shape : ", y_train.shape)
print("X test shape : ", X_test.shape)
print("y test shape : ", y_test.shape)

# Data Preprocessing & PipeLine

y_train = y_train.map({"Y":1, "N":0})
y_test = y_test.map({"Y":1, "N":0})
Numerical_columns = X.select_dtypes(include="number").columns.to_list()
Categorical_columns = X.select_dtypes(exclude="number").columns.to_list()
# print(f"Numerical columns in the data : {Numerical_columns}")
# print(f"Categorical columns in the data : {Categorical_columns}")

## 1. Numerical
numerical_pipeline = Pipeline(steps=[
                    ("selector", DataFrameSelector(Numerical_columns)),
                    ("impute", KNNImputer(n_neighbors=5)),
                    ("scaler", RobustScaler())
                    ])

## 2. categorical
category_pipeline = Pipeline(steps=[
                    ("selector", DataFrameSelector(Categorical_columns)),
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OrdinalEncoder())
                    ])

## 3. Union all together by feature union
all_pipe = FeatureUnion(transformer_list=[
             ("num", numerical_pipeline),
             ("categ", category_pipeline)
            ])

X_train_final = all_pipe.fit_transform(X_train)
X_test_final = all_pipe.transform(X_test)


# Handle Imbalanced Target

oversample = SMOTE() ## Create object
X_train_Smote, y_train_Smote = oversample.fit_resample(X_train_final, y_train)


#  Modeling Using Decision Tree


clf = DecisionTreeClassifier(ccp_alpha=0.01, splitter="best")
clf.fit(X_train_final, y_train)

train_predictions = clf.predict(X_train_final)
test_predictions = clf.predict(X_test_final)

#### `Score Metrics`
train_accuracy = accuracy_score(y_train, train_predictions).round(2)
test_accuracy = accuracy_score(y_test, test_predictions).round(2)
print(f"Accuracy of Training Data : {train_accuracy*100} %")
print(f"Accuracy of Test Data : {test_accuracy*100} %")

train_f1 = f1_score(y_train, train_predictions).round(2)
test_f1 = f1_score(y_test, test_predictions).round(2)
print(f"F1-Score of Training Data : {train_f1*100} %")
print(f"F1-Score of Test Data : {test_f1*100} %")

train_precision = precision_score(y_train, train_predictions).round(2)
test_precision = precision_score(y_test, test_predictions).round(2)
print(f"Precision-Score of Training Data : {train_precision*100} %")
print(f"Precision-Score of Test Data : {test_precision*100} %")

train_recall = recall_score(y_train, train_predictions).round(2)
test_recall = recall_score(y_test, test_predictions).round(2)
print(f"Recall-Score of Training Data : {train_recall*100} %")
print(f"Recall-Score of Test Data : {test_recall*100} %")

cm_test = confusion_matrix(y_test, test_predictions)
matrix = ConfusionMatrixDisplay(confusion_matrix = cm_test, display_labels = [0, 1])
matrix.plot(cmap="Blues")
plt.title("ConfusionMatrix.", weight="bold");


sio.dump(all_pipe, "Model/loan_pipeline.skops")
sio.load("Model/loan_pipeline.skops", trusted=True)