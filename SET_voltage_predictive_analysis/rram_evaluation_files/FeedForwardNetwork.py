#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sys

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchsummary import summary
from io import StringIO


# Device configuration: check if there is a configured GPU available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Fully connected neural network with two hidden layers and dropout layers
class NeuralNet(nn.Module):
    def __init__(self, X_scaled, y_train, parameters):
        super(NeuralNet, self).__init__()
        self.X_train = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        self.y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
        self.input_size = parameters['input_size']
        self.hidden_size = parameters['hidden_size']
        self.output_size = parameters['output_size']
        self.batch_size = parameters['batch_size']
        self.learning_rate = parameters['learning_rate']
        self.num_epochs = parameters['num_epochs']
        self.construct()
        
    def construct(self):  
        self.fc1 = nn.Linear(self.input_size, self.hidden_size) 
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size) 
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size) 
        self.relu3 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.dropout3(out)
        
        out = self.fc4(out)
        return out
    
    # Add this function to compute RMSE
    def rmse_score(self, y_true, y_pred):
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        return rmse

    # Add this function to compute Explained Variance Score
    def evs_score(self, y_true, y_pred):
        evs = 1 - (np.var(y_true - y_pred) / np.var(y_true))
        return evs
        
    # Add this function to compute R-squared
    def r2_score(self, y_true, y_pred):
        y_mean = y_true.mean()
        ss_total = ((y_true - y_mean)**2).sum()
        ss_res = ((y_true - y_pred)**2).sum()
        r2 = 1 - (ss_res / ss_total)
        return r2

    # Add this function to compute MAPE
    def mape_score(self, y_true, y_pred):
        mape = np.mean(np.abs((y_true - y_pred) / (y_true.astype(float) + 1e-10))) * 100
        return mape
    
    def train_model(self):
        # Define the loss function (MSE) and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.00001)
        # Train the model
        loss_values = []  # Store the loss values at each epoch
        for epoch in range(self.num_epochs):
            # Forward pass
            outputs = self(self.X_train)
            loss = criterion(outputs.squeeze(), self.y_train)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_values.append(loss.item())  # Append the current loss value
        
        # Calculate metrics
        r2 = self.r2_score(self.y_train.cpu().numpy(), outputs.detach().cpu().numpy().squeeze())
        mape = self.mape_score(self.y_train.cpu().numpy(), outputs.detach().cpu().numpy().squeeze())
        rmse = self.rmse_score(self.y_train.cpu().numpy(), outputs.detach().cpu().numpy().squeeze())
        evs = self.evs_score(self.y_train.cpu().numpy(), outputs.detach().cpu().numpy().squeeze())
        
        # Store metrics in a dictionary
        metrics = {
            'Loss': loss.item(),
            'R-squared': r2,
            'MAPE': mape,
            'RMSE': rmse,
            'Explained Variance Score': evs
        }

        # Save the model checkpoint
        torch.save(self.state_dict(), '../process_data/model.ckpt')
        
        # Return the dictionary of metrics
        return metrics
    
    def testSample(self,scaled_X_test):
        smp_pred = scaled_X_test.clone().detach()
        with torch.no_grad():  # No gradient computation is needed during inference
            y_pred = self(smp_pred)
        y_pred_numpy = y_pred.cpu().numpy().squeeze()
      
        return y_pred_numpy



def save_summary_as_image(model, file_name, file_format='pdf'):
    
    buffer = StringIO()
    original_stdout = sys.stdout  # Add this line
    sys.stdout = buffer
    summary(model, input_size=(6,))
    sys.stdout = original_stdout
    
    # Save the summary text as a PDF or PNG
    summary_text = buffer.getvalue()
    
    fig = plt.figure(figsize=(10, len(summary_text.split('\n')) * 0.3))
    plt.gca().axis('off')
    plt.text(0, 1, summary_text, fontsize=12, va='top')
    
    plt.savefig(file_name, format=file_format, bbox_inches='tight')
    plt.close(fig)

    


