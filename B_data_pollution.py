import numpy as np
from sklearn.datasets import make_regression
import pandas as pd


def data_pollution(X, n_features, outlier_percentage = 0.1, distribution = 'normal', mean = 0, std = 1):
    """
    @param X: the features ndarray
    @param n_features: number of features
    @param outlier_percentage: the percentage of samples that becomes outliers
    @param distribution: the distribution from which to sample outliers, can be:
        - normal
        - random
        - multinomial
    @param mean: if the distribution is normal, the mean of the distribution
    @param std: if the distribution is normal, the std of the distribution
    @return: the polluted dataframe
    """
    
    # create a dataframe
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    n_samples,n_features = df.shape
    

    # Introduce outliers nelle feature in base alla distribuzione dei dati
    size = int(n_samples * outlier_percentage)

    # random pick #size indices of samples to make them outliers
    outlier_indices = np.random.choice(n_samples, size=size, replace=False)

    """
    Nel mio esempio, ho utilizzato loc=0 e scale=10, 
    il che significa che la media della distribuzione normale è 0 e la deviazione standard è 10. 
    Questo comporta che gli spostamenti degli outliers vengano campionati principalmente vicino a zero, 
    ma possono estendersi fino a circa ±10 (a causa della deviazione standard).
    Ad esempio, se vuoi outliers più estremi, puoi aumentare la deviazione standard (scale). 
    Se vuoi outliers più vicini alla tendenza principale, puoi ridurre la deviazione standard. 
    La media (loc) può essere utilizzata per spostare la distribuzione nel complesso.
    """

    # how much i want to shift the outliers and which distribution to use ?
    if distribution == 'normal':
        outliers_shift = np.random.normal(loc=mean, scale=std, size=(size,n_features))  
    elif distribution == 'random':
        outliers_shift = np.random.uniform(mean, std, size=(size,n_features))
    elif distribution == 'multinomial':
        outliers_shift = np.random.multinomial(size=(size,n_features))

    
    df_polluted = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])

    # create the polluted dataset
    for j,i in enumerate(outlier_indices):
        df_polluted.iloc[i] += outliers_shift[j]

    return df_polluted.values,outlier_indices



