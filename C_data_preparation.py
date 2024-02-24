import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def data_preparation(X,y,n_features):

    """
    @param X: the ndarray to clean
    """
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])

    df['target'] = y

    eps = 0.9  
    min_samples = 5  
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['cluster'] = dbscan.fit_predict(df[['feature_0', 'feature_1', 'feature_2']])

    

    

    clean_df = df[df['cluster'] != -1]

    outlier_indices = df[df['cluster'] == -1].index
    
    # feature_0,feature_1, feature_2, target, cluster
    return clean_df[['feature_0', 'feature_1', 'feature_2']].values, clean_df['target'].values,outlier_indices
    

    
    






