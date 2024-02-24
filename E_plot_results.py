import matplotlib.pyplot as plt
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

your_path = "/Users/franc/Desktop/" # example of path

def mean(results_all):
    list_mean = []
    for res in results_all:
        list_mean.append(res["mean_perf"])
    return list_mean

def distance(results_all):
    list_over = []
    for res in results_all:
        list_over.append(res["distance"])
    return list_over

def speed(results_all):
    list_speed = []
    for res in results_all:
        list_speed.append(res["speed"])
    return list_speed

def generateFigurePerformance(x_axis, xlabel, results_all, title, legend, score,f):

    plt.title(title)
    for i in range(0,len(results_all)):

        mean_perf = mean(results_all[i])

        plt.plot(x_axis, mean_perf, marker='o', label=legend[i], markersize=3)

    plt.xlabel(xlabel)
    plt.ylabel(score)
    plt.legend()
    #plt.ylim(0.1, 2)  # if you want to fix a limit for the y_axis
    plt.savefig("/Users/franc/Desktop/plots/RMSE/" + str(f) + ".jpg", bbox_inches='tight') # if you want to save the figure
    plt.show()

def generateFigureDistance(x_axis, xlabel, results_all, title, legend, score,f):

    plt.title(title)
    for i in range(0,len(results_all)):

        distance_perf = distance(results_all[i])

        plt.plot(x_axis, distance_perf, marker='o', label=legend[i], markersize=3)

    plt.xlabel(xlabel)
    plt.ylabel(score)
    plt.legend()
    #plt.ylim(0.1, 2) # if you want to fix a limit for the y_axis
    plt.savefig("/Users/franc/Desktop/plots/distance/" + str(f) + ".jpg", bbox_inches='tight') # if you want to save the figure
    plt.show()

def generateFigureSpeed(x_axis, xlabel, results_all, title, legend, score,f):

    plt.title(title)
    for i in range(0,len(results_all)):

        speed_perf = speed(results_all[i])

        plt.plot(x_axis, speed_perf, marker='o', label=legend[i], markersize=3)

    plt.xlabel(xlabel)
    plt.ylabel(score)
    plt.legend()
    #plt.ylim(0.1, 2)  # if you want to fix a limit for the y_axis
    #plt.savefig(your_path + title + ".pdf", bbox_inches='tight') # if you want to save the figure
    plt.show()

def plot(x_axis_values, x_label, results, title, algorithms, plot_type,f):

    title = str(title)

    if plot_type == "performance":
        if algorithms[0] == "DecisionTree":
            generateFigurePerformance(x_axis_values, x_label, results, title, algorithms, "f1 weighted",f)
        elif algorithms[0] == "LinearRegressor":
            generateFigurePerformance(x_axis_values, x_label, results, title, algorithms, "RMSE",f)
        else:
            generateFigurePerformance(x_axis_values, x_label, results, title, algorithms, "silhouette",f)

    elif plot_type == "distance train-test": # only for classification & regression
        if algorithms[0] == "DecisionTree":
            generateFigureDistance(x_axis_values, x_label, results, title, algorithms, "f1_train - f1_test",f)
        else:
            generateFigureDistance(x_axis_values, x_label, results, title, algorithms, "RMSE_test - RMSE_train",f)

    else:
        generateFigureSpeed(x_axis_values, x_label, results, title, algorithms, "speed",f)


#6x10 dizionario da cui prendere
def print_table(name, results,filename, metric = "mean_perf"):
    df = pd.DataFrame({
    'LinearRegressor': [item[metric] for item in results[0]],
    'BayesianRidge': [item[metric] for item in results[1]],
    'GPRegressor': [item[metric] for item in results[2]],
    'SVMRegressor': [item[metric] for item in results[2]],
    'KNNRegressor': [item[metric] for item in results[3]],
    'MLPRegressor': [item[metric] for item in results[4]]
    })

    df.index = (df.index * 5) + 5

    df.to_latex(filename, index=True, caption=name)

    

def print_dataset_polluted(X_polluted_list,X_outliers_list,i):
    # Crea i subplot
        fig = make_subplots(
            rows=2, cols=5,
            subplot_titles=("5 % outliers", "10 % outliers", "15 % outliers","20 % outliers","25 % outliers","30 % outliers","35 % outliers","40 % outliers","45 % outliers","50 % outliers"),
            specs=[ [{'type': 'scatter3d'} for _ in range(5)] for _ in range(2)],
            column_widths=[0.2 for _ in range(5)],
            row_heights=[0.5,0.5]
        )
        

        for k in range(2):
            for j in range(5):
                # Crea tracce di tipo go.Scatter3d e aggiungile ai subplots
                df = pd.DataFrame(X_polluted_list[k*5+j], columns=[f'feature_{i}' for i in range(3)])
                scatter1 = go.Scatter3d(x=df['feature_0'], y=df['feature_1'], z=df['feature_2'], mode='markers',opacity=0.5, marker=dict(
                        color=np.where(np.isin(df.index, X_outliers_list[k*5+j]), '#FF6347', 'blue'),
                        size=5
                ))
                fig.add_trace(scatter1, row=k+1, col=j+1)

            # Aggiorna il layout e mostra il plot
        
        
        fig.update_layout(height=700, width=1300, title_text="Experiment {}: polluted datasets".format(i))
        fig.show()


def print_dataset_cleaned(X_polluted_list,X_cleaned_list,i):
    # Crea i subplot
        fig = make_subplots(
            rows=2, cols=5,
            subplot_titles=("5 % outliers", "10 % outliers", "15 % outliers","20 % outliers","25 % outliers","30 % outliers","35 % outliers","40 % outliers","45 % outliers","50 % outliers"),
            specs=[ [{'type': 'scatter3d'} for _ in range(5)] for _ in range(2)],
            column_widths=[0.2 for _ in range(5)],
            row_heights=[0.5,0.5]
        )
        

        for l in range(2):
            for j in range(5):
                # Crea tracce di tipo go.Scatter3d e aggiungile ai subplots
                df_clean = pd.DataFrame(X_cleaned_list[l*5+j], columns=[f'feature_{i}' for i in range(3)])
                df_polluted = pd.DataFrame(X_polluted_list[l*5+j], columns=[f'feature_{i}' for i in range(3)])
                df_outliers = pd.merge(df_polluted, df_clean, how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
                
                not_outlier = go.Scatter3d(
                    x=df_clean['feature_0'], 
                    y=df_clean['feature_1'], 
                    z=df_clean['feature_2'], 
                    mode='markers',
                    marker=dict(
                        color="blue",
                        size=5
                    ),
                    opacity=0.5,
                    name="not-outlier"
                )


                outlier = go.Scatter3d(
                    x=df_outliers['feature_0'], 
                    y=df_outliers['feature_1'], 
                    z=df_outliers['feature_2'], 
                    mode='markers',
                    marker=dict(
                        color="#FF6347",
                        size=5
                    ),
                    opacity=0.5,
                    name="outlier"
                )

                fig.add_trace(not_outlier, row=l+1, col=j+1)
                fig.add_trace(outlier, row=l+1, col=j+1)


                

            # Aggiorna il layout e mostra il plot
        
        
        fig.update_layout(height=700, width=1300, title_text="Experiment {}: Datasets with percentage of outliers".format(i))
        fig.show()

def print_precision(results,i):

    plt.title("Experiment {}, outliers identification precision".format(i))
    
    plt.plot([j/10 for j in range(50,510,50)], results,label= 'precision', marker='o', markersize=3)

    plt.xlabel("outlier percentage (%)")
    plt.ylabel("precision")
    plt.legend()
    #plt.ylim(0.1, 2)  # if you want to fix a limit for the y_axis
    plt.savefig("/Users/franc/Desktop/plots/outliers/" + str(i) + ".jpg", bbox_inches='tight') # if you want to save the figure
    plt.show()

        
def calcola_precision(outliers_identificati, outliers_effettivi):
    # Calcola il numero di veri positivi
    true_positives = len(set(outliers_identificati) & set(outliers_effettivi))

    # Calcola il numero di falsi positivi
    false_positives = len(set(outliers_identificati) - set(outliers_effettivi))

    # Calcola la precisione
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0

    return precision
