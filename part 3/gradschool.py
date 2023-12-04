import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from rerf.rerfClassifier import rerfClassifier

# Read the CSV file into a DataFrame
df = pd.read_csv('graduateschool.csv')

X = df[["GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR", "CGPA", "Research"]]
y = df["Chance of Admit"]

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3
)  # 70% training and 30% test

# Create a Gaussian Classifier
clf = rerfClassifier(n_estimators=100)

print(clf)

# Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))