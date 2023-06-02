import numpy as np
import pandas as pd
import re
import math

class Pipeline:

    def __init__(self, file):
        self.file = file
        # stores the raw data
        self.dfI = pd.read_csv(file,index_col=0)
        
        
        
        self.num_rows = self.dfI.shape[0]
        self.num_columns = self.dfI.shape[1]
        # stores the properly formatted data
        

        self.clean_data()
        self.dfF = self.dfI.copy()
        
        
        
        
        self.gen_dfF()
        print(self.dfF)
        self.dfF.to_csv("training2.csv")
       
    # WL at index 8 and 17
    def calculate_wl(self, column):
        
        #switch row and c
        #record1 = self.dfI.iloc[column][int(((self.num_rows-1)/2)-1)]
        record1 = self.dfI.iloc[int(((self.num_rows-1)/2)-1)][column]
        record2 = self.dfI.iloc[self.num_rows-2][column]
        self.dfI.iloc[int(((self.num_rows-1)/2)-1)][column] = self.format_wl(record1)
        self.dfI.iloc[self.num_rows-2][column] = self.format_wl(record2)

        
        

    def format_wl(self, record):
        pattern = r"\b\d+-\d+-\d+\b"
        print(record)
        record = re.findall(pattern, record)
        record=str(record[0]).split("-")
        
        if int(record[1]) == 0:
            return 100.0
        else:
            wl = (float(record[0]) + float(record[2]) * .25) / float(record[1])
            return wl
    
    def clean_data(self):

        for c in range(0, self.num_columns):
            print(c)
            self.calculate_wl(c)
            for r in range(0, self.num_rows-1):
                # replace rs with c and numrows with numcolumns 
                if r == (((self.num_rows-1)/2)-1) or r == (self.num_rows-2):
                    continue
                else:
                    self.dfI.iloc[r][c] = self.remove_labels(self.dfI.iloc[r][c])
        
                    
                    
    # removes the labels from the raw data
    def remove_labels(self,s):
        stat = float(re.findall(r"[-+]?\d*\.\d+|\d+", str(s))[0])
        
        return stat
#---------------------------dfF functions--------------------------------------    
    #use indexing to get the entire row
    def z_score(self, arr, x):
        arr = np.array(arr).astype(float)
        stdev = np.std(arr)
        mean = np.mean(arr)
        return (x - mean) / stdev

    def get_fight_outcome(self):
        outcomes = []
        #change shape to 0
        for c in range(self.dfI.shape[0]):
            #change to iloc [r][numc]
            if self.dfI.iloc[c][self.num_rows].upper() == "W":
                outcomes.append(1)
            else:
                outcomes.append(0)
        self.dfF.append(outcomes)
        
    
    def riemann_sum(self, z):
        lower_bound = -10
        partitions = 1000
        dX = (z - lower_bound) / partitions
        r_sum = 0
        x = lower_bound 
        
        for i in range(0, partitions):
            r_sum += self.standard_distribution(x + dX) * dX
            x += dX
        return r_sum
            
    def standard_distribution(self, z):
        return 1 / math.sqrt(2 * math.pi) * math.e ** ((-z ** 2)/2)
 
    def gen_dfF(self):
        
        for i in range(0, int((self.num_rows-1)/2)):        
        
            f1 = np.array(self.dfI.iloc[i])
            f2 = np.array(self.dfI.iloc[i+int((self.num_rows-1)/2)])
            arr = np.concatenate((f1, f2), axis=None)

            f1F = np.zeros(int(len(arr)/2))
            f2F = np.zeros(int(len(arr)/2))
            for x in range(len(arr)):
                y = self.riemann_sum(self.z_score(arr, arr[x]))
                if x < len(arr)/2:
                    f1F[x] = y
                else:
                    f2F[x-int(len(arr)/2)] = y
            self.dfF.iloc[i] = f1F
            self.dfF.iloc[i+int((self.num_rows-1)/2)] = f2F
        for c in range(self.num_columns):
            outcome = self.dfI.iloc[self.num_rows-1][c]
            if outcome.upper() == "W":
                self.dfF.iloc[self.num_rows-1][c] = 1
            else:
                self.dfF.iloc[self.num_rows-1][c] = 0
        