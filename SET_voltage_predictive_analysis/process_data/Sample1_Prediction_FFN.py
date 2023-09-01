#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import nbimporter
from getKDE import getKDE
from FeedForwardNetwork import NeuralNet, save_summary_as_image
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import seaborn as sns
import matplotlib.pyplot as plt

# Device configuration: check if there is a configured GPU available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[2]:


# Load the dataframes
setVoltageDf = pd.read_csv('process_data/setValues_KDE.csv')
setVoltageDf.set_index('Sample', inplace=True)
display(setVoltageDf)


# # Isolate the Fabrication Sample

# In[3]:


smp1 = setVoltageDf.loc['Sample_1']
smp1_X = smp1.drop('Set_voltage_kde',axis=1)
smp1_y = setVoltageDf['Set_voltage_kde']
display(smp1)
smp1.to_csv('smp1.csv')


# In[4]:


setVoltageDf = setVoltageDf.drop(index='Sample_1')
idx = setVoltageDf.index.unique()
s_idx = 'Sample_1'
if s_idx in idx:
    raise ValueError('Sample in Training set')


# In[5]:


X = setVoltageDf.drop('Set_voltage_kde',axis=1)
y = setVoltageDf['Set_voltage_kde']


# In[63]:


# Standard Scaling (Z-score normalization)
standard_scaler = StandardScaler()
X_scaled = standard_scaler.fit_transform(X_train)


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
for _ in range(100):
    model = NeuralNet(X_scaled,y_train,parameters).to(device)
    model.train_model()
    scaled_smp1_X = standard_scaler.transform(smp1_X)
    smp1_pred_FFN = model.testSample(scaled_smp1_X)
    ffn_predictions.append(smp1_pred_FFN)
    

def average_line(data,model_name,window_size):
    
    # Calculate the cumulative sum of values at each index
    cumulative_sum = np.sum(data, axis=0)
    
    # Calculate the average by dividing the cumulative sum by the number of arrays
    average_line = cumulative_sum / len(data)

    # Plotting the individual lines
    for line in data:
        plt.plot(line)
    
    #Calculate ROlling Average
    """rolling_avg = np.zeros(len(average_line))
    for i in range(window_size, len(average_line) + 1):
        rolling_avg[i - 1] = np.mean(average_line[i - window_size:i])"""

    # Plotting the average line
    plt.plot(average_line, linewidth=2, color='red', label=model_name+'_Average')

    # Customize the plot
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.legend()

    # Display the plot
    plt.show()
    plt.savefig('Figures/AVG'+model_name+'_Sample_1.jpg',dpi=300)
    return average_line


smp1_pred_ffn['Set_voltage_kde'] = average_line(ffn_predictions,"FFN",window_size)


# In[64]:


save_summary_as_image(model, 'Figures/model_summary2.png', file_format='png')


# In[66]:


smp1_pred_FFN


# In[67]:


ax = plt.gca()
smp1_pred_FFN.plot(kind='line',x='SET Voltage',y='Set_voltage_kde', color='#C20078', ax=ax,label="FFN", linestyle='--')
smp1.plot(kind='line',x='SET Voltage',y='Set_voltage_kde', color='#069AF3', ax=ax,label="True")
plt.rcParams["figure.figsize"] = (10,6.5)
plt.rcParams.update({'font.size': 14})
plt.ylabel("Density")
#plt.legend(prop={'size': 16})
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.title('Sample A')
plt.tight_layout()
plt.savefig('Figures/Leave_One_Out_Sample_1.jpg',dpi=300)
plt.show()


# In[70]:


ax = plt.gca()
smp1_pred_FFN.plot(kind='line',x='SET Voltage',y='Set_voltage_kde', color='Orange', ax=ax,label="FFN")
smp1.plot(kind='line',x='SET Voltage',y='Set_voltage_kde', color='#069AF3', ax=ax,label="True")
#smooth_smp_FFN.plot(kind='line',x='SET Voltage',y='Set_voltage_kde', color='#FFD700', ax=ax,label="FFN")
plt.rcParams["figure.figsize"] = (10,6.5)
plt.rcParams.update({'font.size': 14})
plt.ylabel("Density")
plt.legend(prop={'size': 16})
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.title('Sample A')
plt.tight_layout()
plt.savefig('Figures/FFN_Sample_1.jpg',dpi=300)
plt.show()


# In[71]:


from frechetdist import frdist
s_true = np.array(smp1['Set_voltage_kde'].values).reshape(-1,1)
s_FFN = np.array(smp1_pred_FFN['Set_voltage_kde'].values).reshape(-1,1)

print("Frechet distance FFN:",frdist(s_FFN,s_true))


# In[72]:


from scipy.stats import wasserstein_distance
s_true = np.array(smp1['Set_voltage_kde'].values).reshape(-1,1).ravel().tolist()
s_FFN = np.array(smp1_pred_FFN['Set_voltage_kde'].values).reshape(-1,1).ravel().tolist()

# Calculate Wasserstein distance
print("Wasserstein distance FFN:", wasserstein_distance(s_FFN, s_true))

