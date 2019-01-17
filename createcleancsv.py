import numpy as np
import pandas as pd
from pandas import DataFrame

df = pd.read_csv('train_processed.csv')

unique = pd.value_counts(df.Id)
print(unique.head())
num_classes = unique.values.shape[0]
print(unique.index)
index_0 = unique.index
print(index_0.shape)
print(len(list(index_0)))
print(num_classes)

newdf = DataFrame(columns=['Image','Id'])
print(newdf)

for i in list(index_0):
    newdf = newdf.append(df[df['Id']==i].head(4),ignore_index=True)

newdf.to_csv('train_processed_1.csv')