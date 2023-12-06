# https://sporf.neurodata.io/reference
import csv
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from rerf.rerfClassifier import rerfClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

RANDOM_SEED = 42

from encoders.regression import KMeansEncoder

# Read the CSV file into a DataFrame
df = pd.read_csv('scryfall/parsedCards.csv')

X = df[['C', 'W', 'B', 'U', 'R', 'G', 'X', 'cost', 'artifact', 'battle', 'creature', 'enchantment', 'instant', 'land', 'planeswalker', 'sorcery', 'tribal', 'oracle', 'id']]


data: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series] = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_SEED
)

(X_train, X_vali, y_train, y_vali) = data

# Setup Encoder for regression
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_vali = scaler.transform(X_vali)

y_train = y_train.values.reshape(-1, 1)
y_vali = y_vali.values.reshape(-1, 1)


encoder = KMeansEncoder(random_state=RANDOM_SEED)
encoder.fit(y_train)
print(encoder)

y_train_encoded = encoder.encode(y_train)
y_vali_encoded = encoder.encode(y_vali)

# Setup gaussian SPORF classifier
clf = rerfClassifier(n_estimators=100, random_state=RANDOM_SEED)
print(clf)

clf.fit(X_train, y_train_encoded.ravel())

# Manual score calculation

y_train_pred_encoded = np.array(clf.predict(X_train)).reshape(-1, 1)
y_train_pred = encoder.decode(y_train_pred_encoded)

y_vali_pred_encoded = np.array(clf.predict(X_vali)).reshape(-1, 1)
y_vali_pred = encoder.decode(y_vali_pred_encoded)

print("Raw Classifier and Encoder")
print("[T]Mean Error:", metrics.mean_absolute_error(y_train, y_train_pred))
print("[T]Accuracy:", metrics.accuracy_score(y_train_encoded, y_train_pred_encoded))

print("[V]Mean Error:", metrics.mean_absolute_error(y_vali, y_vali_pred))
print("[V]Accuracy:", metrics.accuracy_score(y_vali_encoded, y_vali_pred_encoded))

y_train_pred_encoded = np.array(clf.predict_proba(X_train))
y_train_pred = encoder.decode(y_train_pred_encoded)

y_vali_pred_encoded = np.array(clf.predict_proba(X_vali))
y_vali_pred = encoder.decode(y_vali_pred_encoded)

print("Using Probabilities to weight clusters")
print("[T]Mean Error:", metrics.mean_absolute_error(y_train, y_train_pred))
print("[V]Mean Error:", metrics.mean_absolute_error(y_vali, y_vali_pred))
