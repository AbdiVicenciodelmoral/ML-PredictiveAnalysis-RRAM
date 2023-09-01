
import pandas as pd
import numpy as np
import nbimporter
from FeedForwardNetwork import NeuralNet, save_summary_as_image
from sklearn.preprocessing import StandardScaler
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from frechetdist import frdist
from scipy.stats import wasserstein_distance

# Device configuration: check if there is a configured GPU available
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
        # Standard Scaling (Z-score normalization)
        standard_scaler = StandardScaler()
        X_scaled = standard_scaler.fit_transform(X_train)
        
        # Move X and y to the same device as the model
        X_scaled = torch.tensor(X_scaled).float().to(device)  
        y_train = torch.tensor(y_train.values).float().to(device)  

        parameters = {
            "input_size":6, 
            "hidden_size":128, 
            "output_size":1,
            "batch_size": 100,   
            "learning_rate": 0.001,
            "num_epochs": 1500         
        }
        ffn_predictions = []
        window_size = 4
        smp_pred_FFN = smp_X
        for i in range(100):
            print(f'Iteration {i} for Test Sample {sample_label}')
            model = NeuralNet(X_scaled.cpu().numpy(), y_train.cpu().numpy(), parameters).to(device)
            model.train_model()
            # Transform the Test Sample with the same scaler
            scaled_smp_X = standard_scaler.transform(smp_X)
            scaled_smp_X = torch.tensor(scaled_smp_X).float().to(device) 
            smp_pred = model.testSample(scaled_smp_X)
            # Append the predictions to the list
            ffn_predictions.append(smp_pred)
        # Calculate the average line of all 100 predicted distribution lines
        smp_pred_FFN['Set_voltage_kde'] = average_line(ffn_predictions,'FFN',sample_idx,title[sample_idx],False,window_size)
        # Save the data, so we can use it later 
        smp_pred_FFN.to_csv(f'../FFN_Predictions/ffn_predictions_sample_{sample_idx}.csv', index=False)  

        plt.figure()
        ax = plt.gca()
        smp_pred_FFN.plot(kind='line',x='SET Voltage',y='Set_voltage_kde', color='#C20078', ax=ax,label="FFN")
        smp.plot(kind='line',x='SET Voltage',y='Set_voltage_kde', color='#069AF3', ax=ax,label="True")
        plt.rcParams["figure.figsize"] = (10,6.5)
        plt.rcParams.update({'font.size': 14})
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'../Figures/FFN_Sample_{sample_idx}.jpg',dpi=300)
        plt.close()  # Close the figure


        s_true = np.array(smp['Set_voltage_kde'].values).reshape(-1,1)
        s_FFN = np.array(smp_pred_FFN['Set_voltage_kde'].values).reshape(-1,1)
        with open(f'../Distances/FFN_Sample{sample_idx}_distances.txt', 'w') as f:
            print("Frechet distance FFN:",frdist(s_FFN,s_true), file=f)


        s_true = np.array(smp['Set_voltage_kde'].values).reshape(-1,1).ravel().tolist()
        s_FFN = np.array(smp_pred_FFN['Set_voltage_kde'].values).reshape(-1,1).ravel().tolist()


        with open(f'../Distances/FFN_Sample{sample_idx}_distances.txt', 'a') as f:  # Notice the 'a' for append mode
            print("Wasserstein distance FFN:", wasserstein_distance(s_FFN, s_true), file=f)




if __name__ == "__main__":
    main()