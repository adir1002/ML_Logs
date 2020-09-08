import tensorflow as tf
from tensorflow import keras 
import seaborn as sns
from pylab import rcParams
from pandas.plotting import register_matplotlib_converters
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import create_out as co
from sklearn.preprocessing import MinMaxScaler
import dateparser
import statistics 

out=372

df={}
for i in range(1,out+1):
    df[i] = pd.read_csv('out'+str(i)+'.csv', parse_dates=['Date'], index_col='Date')
    
out_to_keep = []
    
for i in range(1,out+1):
    if(statistics.variance(df[i].iloc[:,0]) != 0.0):
        out_to_keep.append(i)

new_df = {}
for i in out_to_keep:
    new_df[i] = df[i]
    
for i in out_to_keep:
    scaler = MinMaxScaler()
    scaler = scaler.fit(new_df[i][['Counter']])
    new_df[i]['Logs'] = scaler.transform(new_df[i][['Counter']])
for i in out_to_keep:
    print(i, statistics.variance(new_df[i].iloc[:,1]))
    print(i, "Skewness: %f" % new_df[i].iloc[:,0].skew())
    print(i, "Kurtosis: %f" % new_df[i].iloc[:,0].kurt())
    
    
    
    
    
    