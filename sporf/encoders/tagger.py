import pandas as pd
import numpy as np
from rerf.rerfClassifier import rerfClassifier
from xgboost import XGBClassifier


class MultiClassRerfTagger:
    models: list[rerfClassifier] = []
    num_models: int
    def __init__(self, n_estimators=100, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, Y:pd.DataFrame):
        self.num_models = len(Y.columns)
        for i in range(self.num_models):
            label = Y.iloc[:, i:i+1]
            new_model = rerfClassifier(n_estimators=self.n_estimators, random_state=self.random_state)
            new_model.fit(X, label.values.ravel())
            self.models.append(new_model)
    
    def predict(self, X: pd.DataFrame):
        predictions = []
        
        for model in self.models:
            predictions.append(model.predict(X))
        return np.column_stack(predictions)

class MultiClassXGTagger:
    models: list[XGBClassifier] = []
    num_models: int
    def __init__(self, n_estimators=100, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, Y:pd.DataFrame):
        self.num_models = len(Y.columns)
        for i in range(self.num_models):
            label = Y.iloc[:, i:i+1]
            new_model = XGBClassifier(n_estimators=self.n_estimators, random_state=self.random_state)
            new_model.fit(X, label.values.ravel())
            self.models.append(new_model)
    
    def predict(self, X: pd.DataFrame):
        predictions = []
        
        for model in self.models:
            predictions.append(model.predict(X))
        return np.column_stack(predictions)