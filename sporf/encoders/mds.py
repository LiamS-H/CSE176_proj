from sklearn.manifold import MDS
import numpy as np
import matplotlib.pyplot as plt


class MDSEmbedding:
    labels: list[str]
    def __init__(self, n_dimensions):
        self.n_dimensions = n_dimensions
        self.mds_model = MDS(n_components=n_dimensions, dissimilarity='precomputed', random_state=42)

    def encode(self, input_strings):
        self.labels: list[str] = input_strings
        self.dissimilarity_matrix = self._calculate_dissimilarity_matrix(input_strings)
        encoded_data = self.mds_model.fit_transform(self.dissimilarity_matrix)
        return encoded_data

    def _calculate_dissimilarity_matrix(self, data):
        n_samples = len(data)
        dissimilarity_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            dissimilarity_matrix[i, :] = np.array([self._string_distance(data[i], data[j]) for j in range(n_samples)])

        return dissimilarity_matrix

    def _string_distance(self, s1, s2):
        # smith waterman algorithm
        n = len(s1)
        m = len(s2)
        matrix = np.zeros((n + 1, m + 1))

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if s1[i - 1] == s2[j - 1]:
                    matrix[i][j] = matrix[i - 1][j - 1] + 1
                else:
                    matrix[i][j] = max(matrix[i - 1][j], matrix[i][j - 1])

        return 1 - matrix[n][m] / max(len(s1), len(s2))