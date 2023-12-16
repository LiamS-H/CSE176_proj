#%%

# https://sporf.neurodata.io/reference
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from rerf.rerfClassifier import rerfClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from xgboost import XGBClassifier

RANDOM_SEED = 42

from encoders.regression import KMeansEncoder

# Read the CSV file into a DataFrame
df = pd.read_csv('gradschool/graduateschool.csv')

X = df[["GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR", "CGPA", "Research"]]
y = df["Chance of Admit"]
listOfClassifiers = []

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
# print(encoder)



y_train_encoded = encoder.encode(y_train)
y_vali_encoded = encoder.encode(y_vali)

# Setup gaussian SPORF classifier
hparameter = "n_estimators"
for item in range(0, 20, 1):
    xgboost = XGBClassifier(max_depth=15,n_estimators=item,learning_rate=2,gamma=0#,silent=True,
                            #objective='binary:logistic',booster='gbtree',n_jobs=1,nthread=None,
                            #gamma=0,min_child_weight=1,max_delta_step=0,subsample=1,colsample_bytree=1,
                            #colsample_bylevel=1,reg_alpha=0,reg_lambda=1,scale_pos_weight=1,base_score=0.5,
                            #random_state=0,seed=None,missing=None
                            )
    listOfClassifiers.append(xgboost)

output_score_test = []
output_score_validation = []
output_hparam = []
output_score_test_mean = []
output_score_validation_mean = []
    

for xgboost in listOfClassifiers:
    # print(xgboost)
    xgboost.fit(X_train, y_train_encoded.ravel())

    # Manual score calculation

    y_train_pred_encoded = np.array(xgboost.predict(X_train)).reshape(-1, 1)
    y_train_pred = encoder.decode(y_train_pred_encoded)

    y_vali_pred_encoded = np.array(xgboost.predict(X_vali)).reshape(-1, 1)
    y_vali_pred = encoder.decode(y_vali_pred_encoded)

    print("Raw Classifier and Encoder")
    print("[T]Mean Error:", metrics.mean_absolute_error(y_train, y_train_pred))
    print("[T]Accuracy:", metrics.accuracy_score(y_train_encoded, y_train_pred_encoded))
    output_score_test.append(metrics.accuracy_score(y_train_encoded, y_train_pred_encoded))

    print("[V]Mean Error:", metrics.mean_absolute_error(y_vali, y_vali_pred))
    print("[V]Accuracy:", metrics.accuracy_score(y_vali_encoded, y_vali_pred_encoded))
    output_score_validation.append(metrics.accuracy_score(y_vali_encoded, y_vali_pred_encoded))

    y_train_pred_encoded = np.array(xgboost.predict_proba(X_train))
    y_train_pred = encoder.decode(y_train_pred_encoded)

    y_vali_pred_encoded = np.array(xgboost.predict_proba(X_vali))
    y_vali_pred = encoder.decode(y_vali_pred_encoded)

    print("Using Probabilities to weight clusters")
    print("[T]Mean Error:", metrics.mean_absolute_error(y_train, y_train_pred))
    print("[V]Mean Error:", metrics.mean_absolute_error(y_vali, y_vali_pred))

    output_score_test_mean.append(metrics.mean_absolute_error(y_train, y_train_pred))
    output_score_validation_mean.append(metrics.mean_absolute_error(y_vali, y_vali_pred))
    output_hparam.append(xgboost.get_params()[hparameter])

plt.plot(output_hparam, output_score_test, label="Raw T_accuracy") #test error rate of raw classifier
plt.plot(output_hparam, output_score_validation, label="Raw V_accuracy") #validation error rate of raw classifier

plt.plot(output_hparam, output_score_test_mean, label="Weighted T_error") #test error rate of raw classifier
plt.plot(output_hparam, output_score_validation_mean, label="Weighted V_error") #validation error rate of raw classifier
plt.legend()
plt.title("Error/Accuracy Graph (XGBoost)")
plt.xlabel(hparameter)
plt.ylabel("Error/Accuracy Rate")

plt.savefig('xgboost.png')
# %%
