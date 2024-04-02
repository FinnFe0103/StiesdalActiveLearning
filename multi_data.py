import numpy as np
import math
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import torch

class Dataprep:
    # Upon initialization, load the data, normalize it, and select the initial samples
    def __init__(self, sensors, scaling, initial_samplesize):
        self.X, self.Y = self.load_caselist(sensors)
        self.X, self.Y = self.normalize_data(self.X, self.Y, scaling)
        self.X_selected, self.Y_selected, self.X_pool, self.Y_pool, self.initial_sample = self.initial_sample(self.X, self.Y, initial_samplesize)
        # pd.DataFrame(self.x_selected).to_csv('x_selected.csv', index=False)
        # pd.DataFrame(self.y_selected).to_csv('y_selected.csv', index=False)
        # pd.DataFrame(self.x_pool).to_csv('x_pool.csv', index=False)
        # pd.DataFrame(self.y_pool).to_csv('y_pool.csv', index=False)
        # pd.DataFrame(self.x).to_csv('x.csv', index=False)
        # pd.DataFrame(self.y).to_csv('y.csv', index=False)
    
    def load_caselist(self, sensors): # Load the caselist data from csvs (update later to read directly from the database)
        X = pd.read_csv('_data/caselist.csv')
        Y = pd.read_csv('_data/sim_results.csv')
        Y = np.array(Y[sensors])
        return X.astype(np.float32), Y.astype(np.float32)
    
    def normalize_data(self, x, y, scaling): # Normalize the data using StandardScaler/MinMaxScaler/none
        # Initialize the scaler for x and y
        if scaling == 'Standard':
            scaler_x = StandardScaler()
            scaler_y = StandardScaler()
        elif scaling == 'Minmax':
            scaler_x = MinMaxScaler() 
            scaler_y = MinMaxScaler()
        elif scaling == 'None':
            return x, y
        else:
            raise ValueError('Invalid scaler type')
        
        # Reshape y to have a 2D shape if it's 1D
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        x_normalized = scaler_x.fit_transform(x) # Fit and transform the input features
        y_normalized = scaler_y.fit_transform(y) # Fit and transform the target variable
        #y_normalized = y
        # If y was originally 1D, convert it back
        if y_normalized.shape[1] == 1:
            y_normalized = y_normalized.ravel()
        if x_normalized.shape[1] == 1:
            x_normalized = x_normalized.ravel()
        
        return x_normalized, y_normalized

    def initial_sample(self, x, y, initial_samplesize): # Select the initial samples using K-Medoids clustering (probably needs to be updated using another method)
        indices = np.random.choice(len(x), initial_samplesize, replace=False)
        x_selected = x[indices]
        y_selected = y[indices]
        x_pool = np.delete(x, indices, axis=0)
        y_pool = np.delete(y, indices, axis=0)

        data_known = np.column_stack((x_selected, y_selected))
        data_pool = np.column_stack((x_pool, y_pool))

        return x_selected, y_selected, x_pool, y_pool, indices
    
    def update_data(self, selected_indices): # Update the known and pool data
        self.X_selected = np.append(self.X_selected, self.X_pool[selected_indices], axis=0)
        self.Y_selected = np.append(self.Y_selected, self.Y_pool[selected_indices], axis=0)

        self.X_pool = np.delete(self.X_pool, selected_indices, axis=0)
        self.Y_pool = np.delete(self.Y_pool, selected_indices, axis=0)

        self.initial_sample = np.append(self.initial_sample, selected_indices)

        #print(self.X_selected.shape, self.Y_selected.shape, self.X_pool.shape, self.Y_pool.shape)

def update_data(selected_indices, pool_x, pool_y, known_x, known_y): # 5. Update the known and pool data
    known_x = np.append(known_x, pool_x[selected_indices], axis=0)
    known_y = np.append(known_y, pool_y[selected_indices], axis=0)

    pool_x = np.delete(pool_x, selected_indices, axis=0)
    pool_y = np.delete(pool_y, selected_indices, axis=0)

def load_data(numpy_array, batch_size = 16): # Split the features and target and load the data into a torch DataLoader
    features = numpy_array[:, :-1]#.squeeze()  # All but the last column
    targets = numpy_array[:, -1]    # Last column

    torch_dataset = TensorDataset(torch.tensor(features), torch.tensor(targets).unsqueeze(1))
    torch_loader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=True)
    return torch_loader