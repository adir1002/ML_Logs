# -*- coding: utf-8 -*-

# supress warnings for clean output
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from numpy import percentile
import matplotlib.pyplot as plt
import matplotlib.font_manager
import pandas as pd
import math
import statistics 


# Import all models
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.loci import LOCI
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.sos import SOS
from pyod.models.lscp import LSCP
from pyod.models.cof import COF
from pyod.models.sod import SOD

from sklearn.decomposition import PCA

outliers_fraction = 0.1
clusters_separation = [0]



# initialize a set of detectors for LSCP
detector_list = [LOF(n_neighbors=5), LOF(n_neighbors=10), LOF(n_neighbors=15),
                 LOF(n_neighbors=20), LOF(n_neighbors=25), LOF(n_neighbors=30),
                 LOF(n_neighbors=35), LOF(n_neighbors=40), LOF(n_neighbors=45),
                 LOF(n_neighbors=50)]


random_state = 42
# Define nine outlier detection tools to be compared
classifiers = {
    'Angle-based Outlier Detector (ABOD)':
        ABOD(),
    'Cluster-based Local Outlier Factor (CBLOF)':
        CBLOF(check_estimator=False, random_state=random_state),
    'Feature Bagging':
        FeatureBagging(LOF(n_neighbors=35),
                        random_state=random_state),
    'Histogram-base Outlier Detection (HBOS)': HBOS(),
    'Isolation Forest': IForest(random_state=random_state),
    'K Nearest Neighbors (KNN)': KNN(),
    'Average KNN': KNN(method='mean'),
    # 'Median KNN': KNN(method='median',
    #                   contamination=outliers_fraction),
    'Local Outlier Factor (LOF)':
        LOF(n_neighbors=35),
    # 'Local Correlation Integral (LOCI)':
    #     LOCI(contamination=outliers_fraction),
    'Minimum Covariance Determinant (MCD)': MCD(
                random_state=random_state),
    'One-class SVM (OCSVM)': OCSVM(),
    'Principal Component Analysis (PCA)': PCA(random_state=random_state),
    # 'Stochastic Outlier Selection (SOS)': SOS(
    #     contamination=outliers_fraction),
    'Locally Selective Combination (LSCP)': LSCP(
        detector_list,random_state=random_state),
     # 'Connectivity-Based Outlier Factor (COF)':
     #     COF(n_neighbors=35, contamination=outliers_fraction),
    # 'Subspace Outlier Detection (SOD)':
    #     SOD(contamination=outliers_fraction),
}


# Show all detectors
for i, clf in enumerate(classifiers.keys()):
    print('Model', i + 1, clf)

X_scale = pd.read_csv('after_reduce&scale.csv', parse_dates=['Date'], index_col='Date')

# Fit the models with the generated data and
# compare model performances
df_predicts = pd.DataFrame()
df_scores= pd.DataFrame()
df_predicts_stat=pd.DataFrame()
bestModel=pd.DataFrame()
counter_Anomal=np.array([])
for i, offset in enumerate(clusters_separation):
    np.random.seed(42)

    # Fit the model
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        print()
        print(i + 1, 'fitting', clf_name)
        # fit the data and tag outliers
        clf.fit(X_scale)
        scores_pred = clf.decision_function(X_scale)
        df_scores['score-'+clf_name] = scores_pred
        std_score = df_scores['score-'+clf_name].std()
        mean_score = df_scores['score-'+clf_name].mean()
        y_pred_stat = np.where(df_scores['score-'+clf_name]<mean_score+std_score*2, 0, 1)
        df_predicts_stat[clf_name] = y_pred_stat
        y_pred = clf.predict(X_scale)
        df_predicts[clf_name] = y_pred
        unique, counts = np.unique(y_pred, return_counts=True)
        counter_Anomal = np.append(counter_Anomal, counts[1])
        print(str(dict(zip(unique, counts))))
        

        
df_predicts['final'] = np.where(df_predicts.mean(axis=1)<0.5, 0, 1)
df_predicts_stat['final'] = np.where(df_predicts_stat.mean(axis=1)<0.5, 0, 1)
unique, counts = np.unique(df_predicts['final'], return_counts=True)
print("auto: "+str(dict(zip(unique, counts))))

arr=np.array([])
d = ['ABOD', 'CBLOF','Feature Bagging', 'HBOS','IForest', 'KNN','AKNN','LOF',
     'MCD', 'SVM','PCA','LCSP']
counter=pd.DataFrame(columns=d)
for i in range(len(d)):
    arr= df_predicts.iloc[:,i:i+1].to_numpy().ravel()
    arr= np.where((df_predicts['final']==1) & (arr==1), 1, 0)
    counter = counter.append({str(d[i-3]): np.count_nonzero(arr)}, ignore_index=True)
for i in range(12):
    counter.iloc[0:1,i:i+1]=counter.iloc[:,i:i+1].sum().values[0]
bestModel['Count'] = counter.iloc[0]
bestModel['Percent']=counter.iloc[0]/ counts[1]


unique, counts = np.unique(df_predicts_stat['final'], return_counts=True)
print("stat: "+str(dict(zip(unique, counts))))





