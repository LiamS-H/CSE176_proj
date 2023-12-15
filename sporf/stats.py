import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from encoders.tagreducer import TagEncoder
import matplotlib.pyplot as plt

RANDOM_SEED = 42

dfsize = 10000
sampleSize = 10000
i = 2
df = pd.read_csv(f'scryfall/SCRY10k{i}embed.csv', index_col=0)

sample = df.sample(n=sampleSize, random_state=RANDOM_SEED)

sample  = sample[sample.columns[24:]]
print(sample)
count = sample.sum()
count = count[count >= 300]
count = count.sort_values(ascending=False)


plt.figure()
plot = count.plot(kind='bar', width=0.8, figsize=(15, 5), color='blue')

plot.set_xlabel('Tags')
plot.set_ylabel('Count')
plot.set_title('Tag Distribution')

plt.xticks(rotation=45, ha='right')

for i, v in enumerate(count):
    plot.text(i, v + 0.1, str(v), ha='center', va='bottom', rotation=45)

plt.savefig(f'scryfall/plots/tagdistribution.png')

# row_sums = sample.sum(axis=1)
# row_sum_counts = row_sums.value_counts().sort_index()
# row_sum_counts.plot(kind='bar',figsize=(15, 5))

# plt.xlabel('Number of tags')
# plt.ylabel('Count')
# plt.title('Tags Per Instance')

# plt.savefig(f'scryfall/plots/tagcounts.png')