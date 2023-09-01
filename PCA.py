import pandas as pd
import numpy as np
import re
from matplotlib import pyplot as plt
import seaborn as sb



def clean_data(df):
    df = df.iloc[0:df.shape[0]-1,:]

    for c in range(0, df.shape[1]):
        
        df.iloc[int(((df.shape[0])/2)-1)][c] = format_wl(df.iloc[int(((df.shape[0])/2)-1)][c])
        df.iloc[df.shape[0]-1][c] = format_wl(df.iloc[df.shape[0]-1][c])
        for r in range(0, df.shape[0]-1):
                
            if r == (((df.shape[0])/2)-1) or r == (df.shape[0]-1):
                continue
            else:
                df.iloc[r][c] = remove_labels(df.iloc[r][c])
    return df            

def remove_labels(s):
        stat = float(re.findall(r"[-+]?\d*\.\d+|\d+", str(s))[0])
        return stat

def format_wl(record):
    pattern = r"\b\d+-\d+-\d+\b"
    record = re.findall(pattern, record)
    record=str(record[0]).split("-")
        
    if int(record[1]) == 0:
        return 100.0
    else:
        wl = (float(record[0]) + float(record[2]) * .25) / float(record[1])
        return wl


df = clean_data(pd.read_csv("training.csv",index_col=0)).transpose().astype(float)
zscore = (df - df.mean()) / df.std()
#print(zscore.shape)
#covmatrix = np.cov(zscore)
covmatrix = zscore.cov(numeric_only=True)

sb.heatmap(covmatrix)
plt.show()
