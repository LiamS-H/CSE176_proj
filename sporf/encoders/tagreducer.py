import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

class TagEncoder:
    decode_map={}
    encode_map={}
    columns: int
    def __init__(self):
        return
    
    
    def fit(self, Y: pd.DataFrame): # might break when row is all zeroes
        indices = set()
        sparse_matrix = csr_matrix(Y.values)
        self.columns = Y.columns
        for row in sparse_matrix:
            indice = tuple(row.indices)
            if len(indice) == 0:
                indice = (-1,)
            indices.add(indice)
        
        self.decode_map = {i: tuple_val for i, tuple_val in enumerate(indices)}
        self.encode_map = {tuple_val: i for i, tuple_val in self.decode_map.items()}
        

    def encode(self, Y: pd.DataFrame)->list[int]:
        sparse_matrix = csr_matrix(Y.values)
        encoded = []
        
        for row in sparse_matrix:
            indices = tuple(row.indices)
            if len(indices) == 0:
                indices = (-1,)
            if indices in self.encode_map:
                encoded.append(self.encode_map[indices])
                continue
            if indices[0] in self.encode_map:
                encoded.append(self.encode_map[(indices[0],)])
                continue
            encoded.append(np.nan)

        return encoded
    
    # def decode(self, Y: list[int])->pd.DataFrame:
    #     df = pd.DataFrame(columns=self.columns)
    #     width = len(self.columns)
    #     length = len(Y)
    #     i=0
    #     for val in Y:
    #         i+=1
    #         if i%100 == 0: print(f'{i/length:.2%}')
    #         row = np.zeros(width)
    #         if val != -1:
    #             indices = self.decode_map[val]
    #             row[list(indices)] = 1
    #         df.loc[len(df.index)] = row
    #     return df
    def decode(self, Y):
        width = len(self.columns)
        length = len(Y)
        data = lil_matrix((length, width), dtype=np.int8)

        i = 0
        for val in Y:
            i += 1
            # if i % 100 == 0:
            #     print(f'{i/length:.2%}')

            if val != -1:
                indices = self.decode_map[val]
                data[i - 1, indices] = 1

        df = pd.DataFrame(data.toarray(), columns=self.columns)
        return df