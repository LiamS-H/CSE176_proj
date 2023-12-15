from sklearn.manifold import MDS
import numpy as np


class MDSEmbedding:
    labels: list[str]
    dissimilarity_matrix = None
    def __init__(self, n_dimensions, n_init=4):
        self.n_dimensions = n_dimensions
        self.mds_model = MDS(
            max_iter=1000,
            verbose=2,
            n_jobs=n_init,
            n_components=n_dimensions,
            n_init=n_init,
            dissimilarity='precomputed',
            random_state=42,
            normalized_stress='auto')

    def encode(self, input_strings):
        self.labels: list[str] = input_strings
        if self.dissimilarity_matrix is None:
            self._calculate_dissimilarity_matrix(input_strings)
        encoded_data = self.mds_model.fit_transform(self.dissimilarity_matrix)
        return encoded_data
    
    def _calculate_dissimilarity_matrix(self, data):
        n_samples = len(data)
        dissimilarity_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            if i%100 == 0: print(f'calculating dismat: {i / n_samples:.2%}')
            dissimilarity_matrix[i, :] = np.array([self._string_distance(data[i], data[j]) for j in range(n_samples)])

        self.dissimilarity_matrix = dissimilarity_matrix

    def _string_distance(self, s1: str, s2: str):
        # smith waterman algorithm
        # n = len(s1)
        # m = len(s2)
        # matrix = np.zeros((n + 1, m + 1))

        # for i in range(1, n + 1):
        #     for j in range(1, m + 1):
        #         if s1[i - 1] == s2[j - 1]:
        #             matrix[i][j] = matrix[i - 1][j - 1] + 1
        #         else:
        #             matrix[i][j] = max(matrix[i - 1][j], matrix[i][j - 1])

        # return 1 - matrix[n][m] / max(len(s1), len(s2))
        # simple word inclusions
        s1 = s1.split(" ")
        s2 = s2.split(" ")
        return max(len(s1), len(s2)) - len({word for word in s1 if word in s2})