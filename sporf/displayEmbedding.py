
import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from encoders.tagreducer import TagEncoder
import matplotlib.pyplot as plt

RANDOM_SEED = 42

dfsize = 10000
sampleSize = 1000
i = 2
df = pd.read_csv(f'scryfall/SCRY10k{i}embed.csv')

across_type = ""

sample = df.sample(n=sampleSize, random_state=RANDOM_SEED)
sample = sample[sample["instant"] == 1]
sample = sample[sample["U"] >= 1]
sample = sample.reset_index()


x = sample["e0"]
y = sample["e1"]
attribute = "counterspell"
colors = sample[attribute]
sorted_indices = colors.argsort()
# print(x)
# print(y)
# print(colors)
x = x[sorted_indices]
y = y[sorted_indices]
colors = colors[sorted_indices]

plt.figure()
plt.scatter(x, y, c=colors, cmap='viridis', marker='o', zorder=2)

# Add colorbar for reference
plt.colorbar(label=f'{attribute}')

# Add labels and title
plt.xlabel('e0')
plt.ylabel('e1')
plt.title(f'Scatter Plot of {attribute} across instant|U')
# Show the plot
plt.savefig(f'scryfall/plots/{attribute}n={sampleSize}dim={i}.png')
plt.close()