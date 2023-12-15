import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from rerf.rerfClassifier import rerfClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

def logData(classdata: pd.DataFrame, datalabel="total"):
    count = classdata.sum()
    total = len(classdata)
    print(f'There are {count} {datalabel} intances of {tag_name} ({count/total:.2%})')

def saveConfusion(y_true,y_pred,label=""):
    conf_matrix = metrics.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{tag_name} {label} Confusion Matrix')

    classes = [f'not {tag_name}', tag_name]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes, rotation=90)

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(
                j, i, str(conf_matrix[i, j]),
                fontsize=20,
                horizontalalignment='center', 
                verticalalignment='center')

    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    plt.savefig(f'xgbbin{tag_name}{label}.png')

RANDOM_SEED = 42

sample = pd.read_csv(f'scryfall/SCRY10k5embed.csv', index_col=0)

tag_name = 'animate'

x_feature_indexes = list(range(1, 19)) + list(range(22, 22+5))

X = sample.iloc[:, x_feature_indexes]
y = sample[tag_name]
logData(y)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

c1_count = y_train.sum()
c0_count = len(y_train) - c1_count

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED)
logData(y_val, "validation")
logData(y_test, "test")

n_estimators = 120

boost = XGBClassifier(
    n_estimators=n_estimators,
    random_state=RANDOM_SEED,
    objective='binary:logistic',
    scale_pos_weight=(c0_count/c1_count),
    # scale_pos_weight=3
    
)

boost.fit(X_train, y_train)

y_train_pred = boost.predict(X_train)
y_val_pred = boost.predict(X_val)
y_test_pred = boost.predict(X_test)

train_accuracy = metrics.accuracy_score(y_train, y_train_pred)
val_accuracy = metrics.accuracy_score(y_val, y_val_pred)
test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
print(f'train accuracy: {train_accuracy}')
print(f'val accuracy: {val_accuracy}')
print(f'test accuracy: {test_accuracy}')

saveConfusion(y_train,y_train_pred,"train")
saveConfusion(y_val,y_val_pred,"vali")
saveConfusion(y_test,y_test_pred,"test")