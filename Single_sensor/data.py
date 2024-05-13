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
    def __init__(self, dataset_type, sensor, scaling, initial_samplesize, sampling_method):
        if dataset_type == 'Caselist':
            self.x, self.y= self.load_caselist(sensor)
        elif dataset_type.split('_')[0] == 'Generated':
            dataset_size = int(dataset_type.split('_')[1])
            self.x, self.y = self.generate_data(-10, 10, dataset_size)

        self.x, self.y = self.normalize_data(self.x, self.y, scaling)

        if initial_samplesize > 0:
            self.data_known, self.data_pool = self.initial_sample(self.x, self.y, initial_samplesize, sampling_method)
        else:
            self.data_known, self.data_pool = np.column_stack((self.x, self.y)), np.column_stack((self.x, self.y))

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

        #groups = np.array(x.iloc[:, -1])
        #x = np.array(x.iloc[:, 1:-1])
        y = np.array(y[sensor])
    
        return x.astype(np.float32), y.astype(np.float32)#, groups
    
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

    def initial_sample(self, x, y, initial_samplesize, sampling_method): # Select the initial samples using K-Medoids clustering (probably needs to be updated using another method)
        
        if sampling_method == 'LHC':
            if len(x.shape) <= 1:
                raise ValueError('LHC sampling is only available for multi-dimensional input data. Please use random sampling instead.')
            # scale the x values to the range [0, 1]
            x_scaled, _ = self.normalize_data(x, y, 'Minmax')
            # get the min and max values for every input feature 
            minmax_values = [[np.min(x_scaled[:, i]), np.max(x_scaled[:, i])] for i in range(x_scaled.shape[1])]
            minmax_values = np.array(minmax_values)
            # define the lower and upper bounds for the LHC sampling
            lower_bounds = minmax_values[:, 0]
            upper_bounds = minmax_values[:, 1]
            # generate the LHC samples
            sampler = stats.qmc.LatinHypercube(d=len(minmax_values), optimization="random-cd", seed=42)
            samples = sampler.random(initial_samplesize)
            lower_bounds = minmax_values[:, 0]
            upper_bounds = minmax_values[:, 1]
            samples = stats.qmc.scale(samples, lower_bounds, upper_bounds)

            # get the indices of the closest samples to the LHC samples (LHC normally operating in continuous space, so we need to find the closest samples in the discrete space of the data set)
            indices = []
            for sample in samples:
                distances = np.linalg.norm(x_scaled - sample, axis=1)
                # minimal distance is the maximum similarity
                max_similarity_index = np.argmin(distances)
                index_count = 0
                # Check if the point is already in the matching_indices array
                while max_similarity_index in indices:
                    # Find the index of the second nearest point
                    max_similarity_index = np.argsort(distances)[index_count]
                    index_count += 1
                
                indices.append(max_similarity_index)
            x_selected = x[indices]
            y_selected = y[indices]
            x_pool = np.delete(x, indices, axis=0)
            y_pool = np.delete(y, indices, axis=0)

        else:
            indices = np.random.choice(len(x), initial_samplesize, replace=False)
            x_selected = x[indices]
            y_selected = y[indices]
            x_pool = np.delete(x, indices, axis=0)
            y_pool = np.delete(y, indices, axis=0)

        # if x.ndim == 1:
        #     plt.scatter(x, y, c="blue", label='Data points')
        #     plt.scatter(x_selected, y_selected, c="red", label='Selected Samples')
        #     plt.title('K-Means Clustering with Selected Samples')
        #     plt.xlabel('X variable')
        #     plt.ylabel('Y variable')
        #     plt.legend(fontsize='small')
        #     plt.savefig('_plots/selected.png')
        # else:
        #     pca = PCA(n_components=1)
        #     data_plot = pca.fit_transform(x).squeeze()

        #     # Plotting
        #     plt.scatter(data_plot[:], y, c='lightgray', s=30, label='Data points')
        #     plt.scatter(data_plot[indices], y[indices], c='blue', s=50, label='Selected Samples')
        #     plt.title('K-Means Clustering with Selected Samples')
        #     plt.xlabel('Principal Component 1')
        #     plt.ylabel('Sensor output')
        #     plt.legend(fontsize='small')
        #     plt.savefig('_plots/selected.png')

        data_known = np.column_stack((x_selected, y_selected))
        data_pool = np.column_stack((x_pool, y_pool))

        return data_known, data_pool

def load_data(numpy_array, batch_size = 16): # Split the features and target and load the data into a torch DataLoader
    features = numpy_array[:, :-1]#.squeeze()  # All but the last column
    targets = numpy_array[:, -1]    # Last column

    torch_dataset = TensorDataset(torch.tensor(features), torch.tensor(targets).unsqueeze(1))
    torch_loader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=True)
    return torch_loader