import pandas as pd
import numpy as np


df = pd.read_csv("C:/Users/gocch/Desktop/NeuralNetwork/training3.csv",index_col=0)
df = df.sample(frac=1,axis=1)
print(df)
df.to_csv("shuffledRevisedData.csv")