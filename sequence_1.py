import numpy as np
import pandas as pd
import os
import shutil

df = pd.read_csv('train_processed.csv')
df_train = df.sample(frac=0.9,random_state=0)
print(df_train.index)
df_test = df.drop(df_train.index)
print(df_test.shape)
# np.arange(len(df)*0.9)

for i in df_test['Image']:
    shutil.move('../home/lchn_guo/projects/WhalesServer/generated_train/test/' + str(i) , '../home/lchn_guo/projects/WhalesServer/generated_train/'+ str(i) )

#shutil.move('white_1.jpg', './New_data/move_1.jpg')