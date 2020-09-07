import pandas as pd
import matplotlib.pyplot as plt

def create_file(index_quary=1,filename='out.csv'):
    df=pd.read_csv('a.csv')
    df_t=df.T
    index_t=df_t.index[1:]
    data_t=list(df_t[[index_quary-1]][1:].values)
    quary= str(df_t[[index_quary-1]].values[0:1])
    
    X=pd.DataFrame(data=data_t,columns=['Counter'])
    X=X.set_index(index_t)
    X.index.names = ['Date']
    X.to_csv(filename,index=True)
    # X_plt=X.iloc[0:20,:]
    # plt.figure(1)
    # plt.plot(X_plt, label='Logs - every 15 sec')
    # plt.legend();
    return quary

quary=create_file()
    
    
