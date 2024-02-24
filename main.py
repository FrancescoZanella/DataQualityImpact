import pandas as pd
import numpy as np
import random


from A_data_collection import make_dataset_for_regression
from B_data_pollution import data_pollution
from C_data_preparation import data_preparation
from D_data_analysis import  regression
from E_plot_results import *


# LIST OF ALGORITHMS FOR  REGRESSION  
REGRESSION_ALGORITHMS = ["LinearRegressor","BayesianRidge","GPRegressor","SVMRegressor","KNNRegressor","MLPRegressor"]


SEED = 2023

if __name__ == '__main__':

    print("Main ...")
    
    
    n_features = 3
    n_experiments = 10


    for i in range(n_experiments):
        print("Experiment {}".format(i))

        
        # A: DATA COLLECTION
        X, y = make_dataset_for_regression(n_samples=1000, n_features=n_features, n_informative=3, n_targets=1, bias=10.0, effective_rank=None, tail_strength=0.5, noise=10.0, seed=2023)
        
        
        # B: DATA POLLUTION
        # for each experiment the outlier distance grows
        # from the X we build ten different versions of the dataset with a different percentage of outliers
        X_polluted_list = []
        X_outliers_list = []
        for j in range(5,51,5): 
            a,b = data_pollution(X=X.copy(),n_features=n_features, outlier_percentage=j/100,distribution="random",mean = -10 - i * 10, std = 10 + i*10)
            X_polluted_list.append(a)
            X_outliers_list.append(b)


        ### PRINT THE DATASETS POLLUTED
        print_dataset_polluted(X_polluted_list,X_outliers_list,i)
        
        # D: DATA ANALYSIS ON POLLUTED DATASETS
        results_for_each_algorithm_polluted= []
        for algorithm in REGRESSION_ALGORITHMS:
            results_for_each_percentage_polluted = []
            for d_polluted in X_polluted_list:
                result = regression(d_polluted, y, algorithm, SEED)
                results_for_each_percentage_polluted.append(result)

            results_for_each_algorithm_polluted.append(results_for_each_percentage_polluted)
        

        
        
        # E: PLOT RESULTS ON POLLUTED DATASET
        plot(x_axis_values= [ j for j in range(5,51,5)],x_label="Outlier percentage (%)", results=results_for_each_algorithm_polluted, title="Performance polluted datasets", algorithms=REGRESSION_ALGORITHMS, plot_type="performance",f=i)
        plot(x_axis_values= [ j for j in range(5,51,5)],x_label="Outlier percentage (%)", results=results_for_each_algorithm_polluted, title="Distance train-test polluted datasets", algorithms=REGRESSION_ALGORITHMS, plot_type="distance train-test",f=i)

        
        
        # Create a latex table to later visualization
        print_table(name = "Experiment: {}, mean_perf on polluted dataset".format(i),filename= "{} mean pol.tex".format(i),results = results_for_each_algorithm_polluted,metric="mean_perf")
        print_table(name = "Experiment: {}, distance train-test on polluted dataset".format(i),filename= "{} distance pol.tex".format(i),results = results_for_each_algorithm_polluted,metric="distance")
            

        
        
        

        
        
        
        # C: DATA PREPARATION
        # clean the dataset
        X_cleaned_list = []
        y_cleaned_list = []
        out_list = []
        for h in range(10):
            a,b,c = data_preparation(X_polluted_list.copy()[h],y.copy(),n_features=3)
            X_cleaned_list.append(a)
            y_cleaned_list.append(b)
            out_list.append(c)

        # analyze outlier detection precision
        precision = [calcola_precision(out_list[k],X_outliers_list[k])*100 for k in range(10)]
        print_precision(precision,i)
        print("Experiment {}: MAP= {}".format(i,sum(precision)/len(precision)))

        
        # D: DATA ANALYSIS ON THE CLEANED DATASETS (re-compute the data analysis evaluation on prepared dataset)
        results_for_each_algorithm_cleaned= []
        for algorithm in REGRESSION_ALGORITHMS:
            results_for_each_percentage_cleaned = []
            for h in range(len(X_cleaned_list)):
                result = regression(X_cleaned_list[h], y_cleaned_list[h], algorithm, SEED)
                results_for_each_percentage_cleaned.append(result)

            results_for_each_algorithm_cleaned.append(results_for_each_percentage_cleaned)
        
        ### PLOT THE DATASETS CLEANED
        print_dataset_cleaned(X_polluted_list,X_cleaned_list,i)
        
        # E: PLOT RESULTS ON CLEANED DATASET
        plot(x_axis_values= [ j for j in range(5,51,5) ],x_label="Outlier percentage (%)", results=results_for_each_algorithm_cleaned, title="Performance cleaned datasets", algorithms=REGRESSION_ALGORITHMS, plot_type="performance",f=str(i)+"clean")
        plot(x_axis_values= [ j for j in range(5,51,5) ],x_label="Outlier percentage (%)", results=results_for_each_algorithm_cleaned, title="Distance train-test cleaned datasets", algorithms=REGRESSION_ALGORITHMS, plot_type="distance train-test",f=str(i)+"clean")

        # Create a latex table to later visualization
        print_table(name = "Experiment: {}, mean_perf on cleaned dataset".format(i),filename= "{} mean cleaned.tex".format(i),results=results_for_each_algorithm_cleaned,metric="mean_perf")
        print_table(name= "Experiment: {}, distance train-test on cleaned dataset".format(i),filename= "{} distance cleaned.tex".format(i),results = results_for_each_algorithm_cleaned,metric="distance")
        
       
        
        
        

        

    
    
    
    




    
    
 