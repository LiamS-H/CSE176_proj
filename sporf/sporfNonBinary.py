import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import matplotlib.pyplot as plt
from collections import Counter

from encoders.tagreducer import TagEncoder
from rerf.rerfClassifier import rerfClassifier

def saveConfusion(y_true,y_pred,label=""):
    plt.figure()
    conf_matrix = metrics.confusion_matrix(y_true, y_pred)
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Top {num_tags} {label} tags Confusion Matrix')

    # classes = [encoder.columns]
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes)
    # plt.yticks(tick_marks, classes, rotation=90)

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, str(conf_matrix[i, j]), horizontalalignment='center', verticalalignment='center')

    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    plt.savefig(f'sporfnb{num_tags}{label}.png')

RANDOM_SEED = 42

sample = pd.read_csv(f'scryfall/SCRY10k5embed.csv', index_col=0)

tags  = sample[sample.columns[22+5:]]
count = tags.sum()
count = count[count >= 1000]
count = count[count <= 1200]
tags = list(count.index)
num_tags = len(tags)
print(f"Selecting {num_tags} tags.")

x_feature_indexes = list(range(1, 19)) + list(range(22, 22+5))

X = sample.iloc[:, x_feature_indexes]
y = sample[tags]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED)

encoder = TagEncoder()
encoder.fit(y_train)
class_count = len(encoder.decode_map)
print(f"Encoding Combinations to {class_count} classes")

y_train_encoded = encoder.encode(y_train)
y_val_encoded = encoder.encode(y_val)
y_test_encoded = encoder.encode(y_test)

# remove labels unique to vali and test
y_val["labels"] = y_val_encoded
y_val = y_val.dropna()
X_val = X_val.loc[y_val.index]
y_val_encoded = y_val["labels"]
y_val = y_val.iloc[:, :-1]

y_test["labels"] = y_test_encoded
y_test = y_test.dropna()
X_test = X_test.loc[y_test.index]
y_test_encoded = y_test["labels"]
y_test = y_test.iloc[:, :-1]

print(Counter(y_train_encoded).most_common()[:10])
print(Counter(y_val_encoded).most_common()[:10])
print(encoder.decode([Counter(y_train_encoded).most_common()[0][0]]))

n_estimators = 10

clf = rerfClassifier(
    n_estimators=n_estimators,
    random_state=RANDOM_SEED,
)

clf.fit(X_train, y_train_encoded)

y_train_pred_encoded = clf.predict(X_train)
y_val_pred_encoded = clf.predict(X_val)
y_test_pred_encoded = clf.predict(X_test)

y_train_pred = encoder.decode(y_train_pred_encoded)
y_val_pred = encoder.decode(y_val_pred_encoded)
y_test_pred = encoder.decode(y_test_pred_encoded)

saveConfusion(y_train_encoded, y_train_pred_encoded, "train")
saveConfusion(y_val_encoded, y_val_pred_encoded, "validation")

train_accuracy = metrics.accuracy_score(y_train, y_train_pred)
val_accuracy = metrics.accuracy_score(y_val, y_val_pred)
test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
train_accuracy = metrics.accuracy_score(y_train_encoded, y_train_pred_encoded)
val_accuracy = metrics.accuracy_score(y_val_encoded, y_val_pred_encoded)
test_accuracy = metrics.accuracy_score(y_test_encoded, y_test_pred_encoded)
print(f'train accuracy: {train_accuracy}')
print(f'val accuracy: {val_accuracy}')
print(f'test accuracy: {test_accuracy}')