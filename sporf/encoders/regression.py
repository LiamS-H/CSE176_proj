from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

import numpy as np

def convert_array(arr: list, dim:str="1d"):
    arr = np.array(arr)
    if dim == "1d":
        return arr.ravel() if arr.ndim > 1 else arr.flatten()
    else:
        return arr.reshape(-1, 1) if arr.ndim == 1 else arr


class KMeansEncoder():
    isTrained = False
    mse = None
    def __init__(self, num_clusters=10, random_state=42):
        # Apply k-means clustering
        self.kmeans = KMeans(n_clusters=num_clusters,random_state=random_state, n_init='auto')
        self.clusters = num_clusters

    def __repr__(self):
        if not self.isTrained: return f'<KMeansEncoder untrained k={self.clusters}>'

        return f'<KMeansEncoder MSE={self.mse} k={self.clusters}>'
    
    def fit(self, feature: np.ndarray):
        self.kmeans.fit(feature)
        self.isTrained = True
        feature_encoded = self.encode(feature)
        
        feature_decoded = self.decode(feature_encoded)

        # Evaluate the performance of the autoencoder
        self.mse = mean_squared_error(feature, feature_decoded)
        

    def decode(self, array: np.ndarray):
        if not self.isTrained: raise Exception("Model Not Trained!")
        centroids = self.kmeans.cluster_centers_
        if array.shape[1] == self.clusters:
            probabilities = convert_array(array, "2d")
            decoded = np.einsum('ij,j->i', probabilities, centroids.ravel())
            decoded =  np.array(decoded).reshape(-1,1)
        else:
            cluster_numbers = convert_array(array)
            decoded = centroids[cluster_numbers]
        return np.array(decoded,float)
    
    def encode(self, real_values: np.ndarray):
        if not self.isTrained: raise Exception("Model Not Trained!")
        scaled_values = convert_array(real_values, dim="2d")
        encoded = self.kmeans.predict(scaled_values).reshape(-1, 1)
        return np.array(encoded,int)

    