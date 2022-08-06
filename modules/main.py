import os
import pandas as pd

df = pd.DataFrame()
path = '../data/'
for file in os.listdir(path):
    df_ = pd.read_csv(path + file,index_col=0)
    df = pd.concat([df,df_])

print(df.shape)