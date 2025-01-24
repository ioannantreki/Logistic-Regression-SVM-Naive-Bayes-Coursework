# -*- coding: utf-8 -*-
"""logistic-regression-svm-naive-bayes"""

from google.colab import drive
drive.mount('/content/drive')

# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Loading the dataset into a pandas DataFrame
df = pd.read_csv('/content/drive/MyDrive/heart.csv',delimiter=',')

# Splitting features (X) and target variable (y)
X = df.drop('target', axis=1)  # Features are all columns except 'target'
y = df['target']  # Target variable is the 'target' column

# 1. Prepare dataset (10 points):

# Splitting the data into train and test sets using the ratio of 90% training - 10% test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=22)

# 2. Train a logistic regression model (LR) (25 Points):

# Instantiating the logistic regression model object
lr_model = LogisticRegression(max_iter=1000)

# Training the model on the training data
lr_model.fit(X_train, y_train)

# Testing
lr_pred = lr_model.predict(X_test)

# 3. Train a Support Vector Machine Classifier (SVM) (25 Points):

# Training SVM with different C values
C_values = [1, 10, 20]

for C in C_values:

    # Linear kernel
    svm_linear = SVC(kernel='linear', C=C)
    svm_linear.fit(X_train, y_train)
    svm_linear_pred = svm_linear.predict(X_test)

    # RBF kernel
    svm_rbf = SVC(kernel='rbf', C=C, gamma='auto')
    svm_rbf.fit(X_train, y_train)
    svm_rbf_pred = svm_rbf.predict(X_test)

# 4. Train a Naive Bayes Classifier (NB) (20 Points):

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)

# 5. Evaluate the results (15 Points):

# Defining the evaluate_model function
def evaluate_model(name, y_true, y_pred):
    print(f"{name} Accuracy: {accuracy_score(y_true, y_pred)}")
    print(f"{name} Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}\n")

# Printing the evaluation results

evaluate_model("Logistic Regression", y_test, lr_pred)
print("-" * 50)
print()

for C in C_values:
  evaluate_model(f"SVM Linear (C={C})", y_test, svm_linear_pred)
  evaluate_model(f"SVM RBF (C={C})", y_test, svm_rbf_pred)
  print("-" * 50)
  print()

evaluate_model("Naive Bayes", y_test, nb_pred)
