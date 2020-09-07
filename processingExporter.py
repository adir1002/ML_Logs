# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
from pylab import rcParams
from pandas.plotting import register_matplotlib_converters
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import create_out as co
from sklearn.preprocessing import StandardScaler
    
RANDOM_SEED = 42
THRESHOLD = 0.65
TIME_STEPS = 1
index_plt=1
index_file= 1
out=1

register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 22, 10

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

for i in range(1,out+1):
    co.create_file(i,'out'+str(i)+'.csv')

df={}
for i in range(1,out+1):
    df[i] = pd.read_csv('out'+str(i)+'.csv', parse_dates=['Date'], index_col='Date')
# df=df.iloc[0:100,:]
# plt.figure(index_plt)
# index_plt +=1
# plt.plot(df, label='Logs - every 15 min')
# plt.legend();
index_plt=1
df_test={}
for i in range(1,out+1):
        
    scaler = StandardScaler()
    scaler = scaler.fit(df[i][['Counter']])
    df[i]['Logs'] = scaler.transform(df[i][['Counter']])  
    def create_dataset(X, y, time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = X.iloc[i:(i + time_steps)].values
            Xs.append(v)
            ys.append(y.iloc[i + time_steps])
        i+=1
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i])
        return np.array(Xs), np.array(ys)
    
    # reshape to [samples, time_steps, n_features]
    X, y = create_dataset(
      df[i][['Logs']],
      df[i].Logs,
      TIME_STEPS
    )
    
    print("X shape: "+str(X.shape) + "y shape: "+str(y.shape))
    
    model = keras.Sequential()
    model.add(keras.layers.LSTM(
        units=64, 
        input_shape=(X.shape[1], X.shape[2])
    ))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.RepeatVector(n=X.shape[1]))
    model.add(keras.layers.LSTM(units=64, return_sequences=True))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=X.shape[2])))
    model.compile(loss='mae', optimizer='adam')
    history = model.fit(
        X, y,
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        shuffle=False
    )  
    
    X_pred = model.predict(X)
    
    train_mae_loss = np.mean(np.abs(X_pred - X), axis=1)
    
    # plt.figure(index_plt)
    # index_plt +=1
    # sns.distplot(train_mae_loss, bins=50, kde=True);
    
    test_mae_loss = np.mean(np.abs(X_pred - X), axis=1)
    
    df_test[i]=pd.DataFrame()
    # df_test = pd.DataFrame(df.take(0, axis=2))   
    df_test[i] = pd.DataFrame(index=df[i][:].index)
    df_test[i]['score'] = X_pred.take(0, axis=2)
    df_test[i]['Logs'] = df[i][:].Logs
    df_test[i]['loss'] = test_mae_loss
    df_test[i]['cluster'] = np.where((df_test[i]['score']<
                                               df_test[i]['score'].mean()+
                                                1.5*df_test[i]['score'].std())
                                  &(df_test[i]['score']>
                                               df_test[i]['score'].mean()-
                                                1.5*df_test[i]['score'].std())
                                               , 0, 1)
    print(df_test[i]['cluster'].value_counts())
    df_test[i]['mean_1std']= df_test[i]['score'].mean()+1.5*df_test[i]['score'].std()
    df_test[i]['mean_1std_minus']= -df_test[i]['mean_1std']
    df_test[i]['threshold'] = THRESHOLD
    df_test[i]['anomaly_thers'] = df_test[i].loss > df_test[i].threshold
    
    # plt.figure(index_plt)
    # index_plt +=1
    # plt.plot(df_test.index, df_test.loss, label='loss')
    # plt.plot(df_test.index, df_test.threshold, label='threshold')
    # plt.xticks(rotation=25)
    # plt.legend()
    
    anomalies_thers = df_test[i][df_test[i].anomaly_thers == True]
    anomalies_thers.head()
    
    # plt.figure(index_plt)
    # index_plt +=1
    # plt.plot(
    #   df[:].index, 
    #   scaler.inverse_transform(df[:].Logs), 
    #   label='Server Logs '
    # );
    
    # sns.scatterplot(
    #   anomalies_thers.index,
    #   scaler.inverse_transform(anomalies_thers.Logs),
    #   color=sns.color_palette()[3],
    #   s=52,
    #   label='anomaly'
    # )
    # plt.xticks(rotation=25)
    # plt.title('with thersold')
    # plt.legend();
    
    
    plt.figure(index_plt)
    index_plt +=1
    plt.plot(df_test[i].index, df_test[i].score, label='cluster score')
    plt.plot(df_test[i].index, df_test[i]['mean_1std'], label='mean + 2 std')
    plt.xticks(rotation=25)
    plt.legend()
    
    anomalies = df_test[i][df_test[i].cluster == 1]
    anomalies.head()
    
    plt.figure(index_plt)
    index_plt +=1
    plt.plot(
      df[i][:].index, 
      scaler.inverse_transform(df[i][:].Logs), 
      label='Server Logs '
    );
    
    sns.scatterplot(
      anomalies.index,
      scaler.inverse_transform(anomalies.Logs),
      color=sns.color_palette()[3],
      s=52,
      label='anomaly'
    )
    plt.xticks(rotation=25)
    plt.title('with mean & 2 std')
    plt.legend();


