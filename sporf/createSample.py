# https://sporf.neurodata.io/reference
import csv
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from rerf.rerfClassifier import rerfClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import matplotlib.pyplot as plt

from scryfall.genTags import loadTagMap


RANDOM_SEED = 42

from encoders.mds import MDSEmbedding
df = pd.read_csv('scryfall/cards.csv')
df['oracle'] = df['oracle'].fillna('')
df.dropna()

# for sampleSize in [10000]:
#     # print("getting array")
#     # dissimilarity_matrix = np.load('dismatrix.npy')
#     # print("array loaded")

#     # sample = df
#     sample = df.sample(n=sampleSize, random_state=RANDOM_SEED)
#     text = sample['oracle']

#     embedding = MDSEmbedding(2)
#     encoded_data = embedding._calculate_dissimilarity_matrix(text.values)
#     sample_matrix = embedding.dissimilarity_matrix
#     np.save(f'dismat{sampleSize}.npy', sample_matrix)

#     for i in [5,2]:
#         text = sample['oracle']
#         embedding = MDSEmbedding(i)
#         embedding.dissimilarity_matrix = sample_matrix
#         encoded_data = embedding.encode(text)
        
#         encoded_data = pd.DataFrame(encoded_data, columns=['e0','e1','e2','e3','e4','e5','e6','e7','e8','e9'][0:i]) 

#         encoded_data.index = sample.index
#         out = pd.concat([sample,encoded_data], axis=1, ignore_index=True)

#         out.to_csv(f'scryfall/{sampleSize}dataset{i}.csv')

tag_map = loadTagMap("scryfall/tagmap.json")
for sampleSize in [10000]:
    for i in [5,2]:
        filestr=f'{sampleSize}dataset{i}.csv'
        sample = pd.read_csv("scryfall/"+filestr)

        otags = set()
        for id in sample["id"]:
            if id not in tag_map: continue
            tags=tag_map[id]
            otags.update(tags)
        tags = sorted(list(otags))
        tagged = pd.DataFrame(columns=tags)

        num_cols = len(tags)

        for index, id in enumerate(sample["id"]):
            if index % 100 ==0: print(f'{index/sampleSize:.2%}')
            current_tags=[]
            if id in tag_map:
                current_tags=tag_map[id]
            
            new_row = [1 if tag in current_tags else 0 for tag in tags]
            tagged.loc[len(tagged)] = new_row

        sample = pd.concat([sample,tagged], axis=1)
        sample.to_csv("scryfall/Tagged"+filestr)