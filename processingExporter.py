# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
from pylab import rcParams
from pandas.plotting import register_matplotlib_converters
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import reduceNscale as rs
 
    
RANDOM_SEED = 42
THRESHOLD = 0.65
TIME_STEPS = 1
index_plt=1
index_file= 1
out=372

register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 22, 10

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


rs.reduce_queries(scale='Standard')
df = pd.read_csv('after_reduce&scale.csv', parse_dates=['Date'], index_col='Date')


for i in df.columns:    
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
      df,
      df,
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
test_mae_loss = np.mean(np.abs(X_pred - X), axis=1)

df_test = pd.DataFrame(index=df.index)
df_test['score'] = X_pred.take(0, axis=2)
df_test['loss'] = test_mae_loss
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


anomalies_thers = df_test[i][df_test[i].anomaly_thers == True]
anomalies_thers.head()


anomalies = df_test[i][df_test[i].cluster == 1]
anomalies.head()

