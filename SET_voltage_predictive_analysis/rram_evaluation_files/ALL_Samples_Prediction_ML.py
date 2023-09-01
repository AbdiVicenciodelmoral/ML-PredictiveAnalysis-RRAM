
import pandas as pd
import numpy as np
import nbimporter
from MLModels import mlModels
from sklearn.preprocessing import StandardScaler
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from frechetdist import frdist
from scipy.stats import wasserstein_distance

# check if there is a GPU available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def average_line(data,model_name,sample_idx,sample_label,smooth,window_size):
    
    # Cumulative sum of values at each element along the rows (axis=0)
    # Creates a row that is the sum of all rows
    cumulative_sum = np.sum(data, axis=0)
    
    # Calculate the average of every element
    # by dividing the cumulative sum by the number of arrays
    average_line = cumulative_sum / len(data)
    # Plotting the individual lines in the data
    # to compare with the average line
    plt.figure()
    for line in data:
        plt.plot(line)
    
    # Calculate ROlling Average
    # This will smooth the average line if specified
    if smooth:
        rolling_avg = np.zeros(len(average_line))
        for i in range(window_size, len(average_line) + 1):
            rolling_avg[i - 1] = np.mean(average_line[i - window_size:i])
    # Plotting the average line
    plt.plot(average_line, linewidth=2, color='red', label=model_name+'_Average')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    #plt.title('Sample ' + sample_label)
    plt.legend()
    plt.savefig(f'../Figures/AVG{model_name}_Sample{sample_idx}.jpg', dpi=300)
    # Ensure to close the figure, so that the proceeding plots don't merge
    plt.close()  
    return average_line


def load_data(sample_idx):
    # Load the dataframes
    setVoltageDf = pd.read_csv('../process_data/setValues_KDE.csv')
    setVoltageDf.set_index('Sample', inplace=True)
    print(setVoltageDf)
    smp = setVoltageDf.loc[f'Sample_{sample_idx}']
    smp_X = smp.drop('Set_voltage_kde',axis=1)
    #smp_y = setVoltageDf['Set_voltage_kde']
    print(smp)
    smp.to_csv(f'../Sample_Data/smp{sample_idx}.csv')
    setVoltageDf = setVoltageDf.drop(index=f'Sample_{sample_idx}')
    idx = setVoltageDf.index.unique()
    s_idx = f'Sample_{sample_idx}'
    if s_idx in idx:
        raise ValueError('Sample in Training set')
    return smp, smp_X, setVoltageDf


def main():
    title= {1:"A", 2:"B", 3:"C", 4:"D", 5:"E", 6:"F", 7:"G", 8:"H"}
    for sample_idx in range(1, 9):
        sample_label = title[sample_idx]
        # Load the dataframes
        smp, smp_X, setVoltageDf = load_data(sample_idx)
        # Separate Features and Target
        X_train = setVoltageDf.drop('Set_voltage_kde',axis=1)
        y_train = setVoltageDf['Set_voltage_kde']
        

        smp_pred_ada = smp_X.copy()
        smp_pred_rfr = smp_X.copy()
        ada_predictions = []
        rfr_predictions = []
        
        window_size = 4


        # For Random Forest Regression
        rfr_metrics_sum = {
                "MSE": 0,
                "RMSE": 0,
                "MAE": 0,
                "MAPE": 0,
                "R2": 0,
                "Explained Variance": 0,
            }

                # For Adaboost Regression
        ada_metrics_sum = {
                "MSE": 0,
                "RMSE": 0,
                "MAE": 0,
                "MAPE": 0,
                "R2": 0,
                "Explained Variance": 0,
            }

        trials = 100
        for i in range(trials):
            print(f'Iteration {i} for Test Sample {sample_label}')
            reg_Models = mlModels()
            metrics = reg_Models.runModels(X_train,y_train,grid_search=False)
            smp_pred_ada['Set_voltage_kde'] = reg_Models.newSample(smp_X,'ada')
            ada_predictions.append(smp_pred_ada['Set_voltage_kde'].values)
            smp_pred_rfr['Set_voltage_kde'] = reg_Models.newSample(smp_X,'rfr')
            rfr_predictions.append(smp_pred_rfr['Set_voltage_kde'].values)
            
            # Add current run's metrics to the sum
            for metric, value in metrics['rfr_metrics'].items():
                rfr_metrics_sum[metric] += value

            for metric, value in metrics['ada_metrics'].items():
                ada_metrics_sum[metric] += value

        
        # Calculate averages
        rfr_metrics_avg = {metric: value / trials for metric, value in rfr_metrics_sum.items()}
        ada_metrics_avg = {metric: value / trials for metric, value in ada_metrics_sum.items()}

        print("\nAverage Random Forest metrics over {} trials:".format(trials))
        for metric, avg in rfr_metrics_avg.items():
            print("{}: {:.5f}".format(metric, avg))

        print("\nAverage AdaBoost metrics over {} trials:".format(trials))
        for metric, avg in ada_metrics_avg.items():
            print("{}: {:.5f}".format(metric, avg))



        smp_pred_ada['Set_voltage_kde'] = average_line(ada_predictions,"ADA",sample_idx,sample_label,True,window_size)
        smp_pred_rfr['Set_voltage_kde'] = average_line(rfr_predictions,"RFR",sample_idx,sample_label,True,window_size)
        
        smp_pred_ada.to_csv(f'../ML_Predictions/ADA_predictions_sample_{sample_idx}.csv', index=False)  
        smp_pred_rfr.to_csv(f'../ML_Predictions/RFR_predictions_sample_{sample_idx}.csv', index=False)  

        plt.figure()
        ax = plt.gca()
        smp_pred_ada.plot(kind='line',x='SET Voltage',y='Set_voltage_kde', color='#4CAF50', ax=ax,label="Adaboost", linestyle=':')
        smp_pred_rfr.plot(kind='line',x='SET Voltage',y='Set_voltage_kde', color='#C20078', ax=ax,label="Random Forest", linestyle='-.')
        smp.plot(kind='line',x='SET Voltage',y='Set_voltage_kde', color='#069AF3', ax=ax,label="True")
        plt.rcParams["figure.figsize"] = (10,6.5)
        plt.rcParams.update({'font.size': 14})
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'../Figures/ML_Sample_{sample_idx}.jpg',dpi=300)
        plt.close() 


        s_true = np.array(smp['Set_voltage_kde'].values).reshape(-1,1)
        s_ada = np.array(smp_pred_ada['Set_voltage_kde'].values).reshape(-1,1)
        with open(f'../Distances/ADA_Sample{sample_idx}_distances.txt', 'w') as f:
            print("Frechet distance ADA:",frdist(s_ada,s_true), file=f)

        s_rfr = np.array(smp_pred_rfr['Set_voltage_kde'].values).reshape(-1,1)
        with open(f'../Distances/RFR_Sample{sample_idx}_distances.txt', 'w') as f:
            print("Frechet distance RFR:",frdist(s_rfr,s_true), file=f)



        s_true = np.array(smp['Set_voltage_kde'].values).reshape(-1,1).ravel().tolist()
        s_ada = np.array(smp_pred_ada['Set_voltage_kde'].values).reshape(-1,1).ravel().tolist()
        with open(f'../Distances/ADA_Sample{sample_idx}_distances.txt', 'a') as f:  
            print("Wasserstein distance ADA:", wasserstein_distance(s_ada, s_true), file=f)

        s_rfr = np.array(smp_pred_rfr['Set_voltage_kde'].values).reshape(-1,1).ravel().tolist()
        with open(f'../Distances/RFR_Sample{sample_idx}_distances.txt', 'a') as f:  
            print("Wasserstein distance RFR:", wasserstein_distance(s_rfr, s_true), file=f)




if __name__ == "__main__":
    main()