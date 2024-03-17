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
    def __init__(self, dataset_type, sensor, scaling, initial_samplesize):
        if dataset_type == 'Caselist':
            self.x, self.y, self.groups = self.load_caselist(sensor)
        elif dataset_type.split('_')[0] == 'Generated':
            dataset_size = int(dataset_type.split('_')[1])
            self.x, self.y = self.generate_data(-10, 10, dataset_size)

        self.x, self.y = self.normalize_data(self.x, self.y, scaling)
        self.known_data, self.pool_data = self.initial_sample(self.x, self.y, initial_samplesize)

    def generate_data(self, start, end, n): # Generate synthetic data
        x = np.linspace(start, end, n)
        sample_mean = [math.sin(i/2) for i in x]
        sample_var = [((abs(start)+abs(end))/2 - abs(i))/16 for i in x]
        y = stats.norm(sample_mean, sample_var).rvs()+10

        # Shuffle the x and y values
        indices = np.random.permutation(n)
        x, y = x[indices], y[indices]
        
        return x.astype(np.float32), y.astype(np.float32)
    
    def load_caselist(self, sensor): # Load the caselist data from csvs (update later to read directly from the database)
        x = pd.read_csv('_data/caselist.csv')
        y = pd.read_csv('_data/sim_results.csv')

        groups = np.array(x.iloc[:, -1])
        x = np.array(x.iloc[:, 1:-1])
        y = np.array(y[sensor])
    
        return x.astype(np.float32), y.astype(np.float32), groups
    
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
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
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

        if x.ndim == 1:
            plt.scatter(x, y, c="blue", label='Data points')
            plt.scatter(x_selected, y_selected, c="red", label='Selected Samples')
            plt.title('K-Means Clustering with Selected Samples')
            plt.xlabel('X variable')
            plt.ylabel('Y variable')
            plt.legend(fontsize='small')
            plt.savefig('_plots/selected.png')
        else:
            pca = PCA(n_components=1)
            data_plot = pca.fit_transform(x).squeeze()

            # Plotting
            plt.scatter(data_plot[:], y, c='lightgray', s=30, label='Data points')
            plt.scatter(data_plot[indices], y[indices], c='blue', s=50, label='Selected Samples')
            plt.title('K-Means Clustering with Selected Samples')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Sensor output')
            plt.legend(fontsize='small')
            plt.savefig('_plots/selected.png')

        known_data = np.column_stack((x_selected, y_selected))
        pool_data = np.column_stack((x_pool, y_pool))

        return known_data, pool_data

def load_data(numpy_array, batch_size = 16): # Split the features and target and load the data into a torch DataLoader
    features = numpy_array[:, :-1]#.squeeze()  # All but the last column
    targets = numpy_array[:, -1]    # Last column

    torch_dataset = TensorDataset(torch.tensor(features), torch.tensor(targets).unsqueeze(1))
    torch_loader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=True)
    return torch_loader