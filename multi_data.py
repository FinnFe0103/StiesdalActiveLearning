import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import torch

class Dataprep:
    # Upon initialization, load the data, normalize it, and select the initial samples
    def __init__(self, sensors, scaling, initial_samplesize):
        self.X, self.Y = self.load_caselist(sensors)
        self.X, self.Y = self.normalize_data(self.X, self.Y, scaling)
        self.X_selected, self.Y_selected, self.X_pool, self.Y_pool, self.initial_sample = self.initial_sample(self.X, self.Y, initial_samplesize)
    
    def load_caselist(self, sensors):
        '''
        Load the caselist data from csv files

        Parameters:
        sensors: list
            The sensors to load data for
        
        Returns:
        X: pd.DataFrame
            The input features
        Y: pd.DataFrame
            The target variable
        '''
        
        X = pd.read_csv('EDA_Preprocessing/caselist.csv')
        Y = pd.read_csv('EDA_Preprocessing/sim_results.csv')
        print(Y.columns)
        Y = np.array(Y[sensors])
        return X.astype(np.float32), Y.astype(np.float32)
    
    def normalize_data(self, x, y, scaling):
        '''
        Normalize the input features and target variable

        Parameters:
        x: np.array
            The input features
        y: np.array
            The target variable
        scaling: str
            The type of scaling to use. Options are 'Standard', 'Minmax', or 'None'
        
        Returns:
        x_normalized: np.array
            The normalized input features
        y_normalized: np.array
            The normalized target variable
        '''

        # Initialize the scaler for x and y
        if scaling == 'Standard':
            scaler_x = StandardScaler()
            scaler_y = StandardScaler() # leave out?
        elif scaling == 'Minmax': 
            scaler_x = MinMaxScaler() 
            scaler_y = MinMaxScaler() # leave out?
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

    def initial_sample(self, x, y, initial_samplesize):
        '''
        Randomly select initial samples from the data
        
        Parameters:
        x: np.array
            The input features
        y: np.array
            The target variable
        initial_samplesize: int
            The number of samples to select
        
        Returns:
        x_selected: np.array
            The selected input features
        y_selected: np.array
            The selected target variable
        x_pool: np.array
            The remaining input features
        y_pool: np.array
            The remaining target variable
        indices: np.array
            The indices of the selected samples
        '''

        indices = np.random.choice(len(x), initial_samplesize, replace=False)
        x_selected = x[indices]
        y_selected = y[indices]
        x_pool = np.delete(x, indices, axis=0)
        y_pool = np.delete(y, indices, axis=0)

        return x_selected, y_selected, x_pool, y_pool, indices
    
    def update_data(self, selected_indices):
        '''
        Update the known and pool data after selecting samples

        Parameters:
        selected_indices: np.array
            The indices of the selected samples
        '''
        self.X_selected = np.append(self.X_selected, self.X_pool[selected_indices], axis=0)
        self.Y_selected = np.append(self.Y_selected, self.Y_pool[selected_indices], axis=0)

        self.X_pool = np.delete(self.X_pool, selected_indices, axis=0)
        self.Y_pool = np.delete(self.Y_pool, selected_indices, axis=0)

        self.initial_sample = np.append(self.initial_sample, selected_indices)

        #print(self.X_selected.shape, self.Y_selected.shape, self.X_pool.shape, self.Y_pool.shape)