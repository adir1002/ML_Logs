# -*- coding: utf-8 -*-
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import statistics 
import errno
    
OUT=372

def create_separated_files(out=OUT):
    try:
        os.makedirs('queries')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    for i in range(1,out+1):
        create_file(i,'queries/query'+str(i)+'.csv')
  
def create_file(index_quary=1,filename='queries/query.csv'):
    df=pd.read_csv('all_queries.csv')
    df_t=df.T
    index_t=df_t.index[1:]
    data_t=list(df_t[[index_quary-1]][1:].values)
    quary= str(df_t[[index_quary-1]].values[0:1])
    
    X=pd.DataFrame(data=data_t,columns=['Counter'])
    X=X.set_index(index_t)
    X.index.names = ['Date']
    X.to_csv(filename,index=True)
    return quary

        
def reduce_queries(scale='none',out=OUT,separately=True):
    df={}
    for i in range(1,out+1):
        df[i] = pd.read_csv('queries/query'+str(i)+'.csv', parse_dates=['Date'],index_col=['Date'])
    #Reduce queries with zero variance
    out_to_keep = []
    for i in range(1,out+1):
        if(statistics.variance(df[i].iloc[:,0]) != 0.0):
            out_to_keep.append(i)
            print(str(i)+'-------'+str(statistics.variance(df[i].iloc[:,0])))         
    new_df = {}
    for i in out_to_keep:
        new_df[i] = df[i]
    if scale=='MinMax':
        for i in out_to_keep:  
            scaler = MinMaxScaler()
            scaler = scaler.fit(new_df[i][['Counter']])
            new_df[i]['Counter'] = scaler.transform(new_df[i][['Counter']])  
    elif scale=='Standard':
        for i in out_to_keep:  
            scaler = StandardScaler()
            scaler = scaler.fit(new_df[i][['Counter']])
            new_df[i]['Counter'] = scaler.transform(new_df[i][['Counter']]) 
    df_full=pd.DataFrame()
    for i in out_to_keep:
        df_full[i]=new_df[i].iloc[:,0]        
    df_full.to_csv('after_reduce&scale.csv',index=True)
        

# create_separated_files(out=372)
# reduce_queries(scale='MinMax')
