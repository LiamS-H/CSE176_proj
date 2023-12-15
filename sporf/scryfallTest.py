# https://sporf.neurodata.io/reference
import csv
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from rerf.rerfClassifier import rerfClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from encoders.tagreducer import TagEncoder

from collections import Counter

RANDOM_SEED = 42


sampleSize = 5000
i = 2

sample = pd.read_csv(f'scryfall/samples/n={sampleSize}/embeddedCards{i}.csv')

sample = sample.apply(pd.to_numeric, errors='coerce')

sample = sample.fillna(0)


features = list(range(3, 19)) + list(range(21, 21 + i))

X = sample.iloc[:, features]

y = sample.iloc[:, 21+i:]

data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] = train_test_split(
    X, y, test_size=0.5, random_state=RANDOM_SEED
)

(X_train, X_vali, y_train, y_vali) = data


encoder = TagEncoder()
encoder.fit(y_train)

y_train_encoded = encoder.encode(y_train)
y_vali_encoded = encoder.encode(y_vali)

y_vali["labels"] = y_vali_encoded


kept_indices = y_vali.dropna().index

y_vali = y_vali.loc[kept_indices]
X_vali = X_vali.loc[kept_indices]

y_vali_encoded = y_vali["labels"]


# Setup Unsupervised Random Forest classifier
clf = rerfClassifier(n_estimators=100, random_state=RANDOM_SEED)

print("xt",len(X_train))
print("yt",len(y_train))

print("xv",len(X_vali))
print("yv",len(y_vali_encoded))

clf.fit(X_train, y_train_encoded)


y_train_pred_encoded = clf.predict(X_train)
y_vali_pred_encoded = clf.predict(X_vali)

y_train_pred = encoder.decode(y_train_pred_encoded)

y_vali_pred = encoder.decode(y_vali_pred_encoded)


train_accuracy = metrics.accuracy_score(y_train_encoded, y_train_pred_encoded)
vali_accuracy = metrics.accuracy_score(y_vali_encoded, y_vali_pred_encoded)

print("[T]Accuracy:", train_accuracy)
print("[V]Accuracy:", vali_accuracy)