import numpy as np
import math
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import torch


class Dataprep:
    # Upon initialization, load the data, normalize it, and select the initial samples
    def __init__(self, dataset_type, sensor, initial_samplesize):

        if dataset_type == 'Caselist':
            self.x, self.y, self.groups = self.load_caselist(sensor)
        elif dataset_type.split('_')[0] == 'Generated':
            dataset_size = int(dataset_type.split('_')[1])
            self.x, self.y = self.generate_data(-10, 10, dataset_size)

        self.x, self.y = self.normalize_data(self.x, self.y)
        self.known_data, self.pool_data = self.initial_sample(self.x, self.y, initial_samplesize)

    # Generate synthetic data
    def generate_data(self, start, end, n):
        x = np.linspace(start, end, n)
        sample_mean = [math.sin(i/2) for i in x]
        sample_var = [((abs(start)+abs(end))/2 - abs(i))/16 for i in x]
        y = stats.norm(sample_mean, sample_var).rvs()

        # Shuffle the x and y values
        indices = np.random.permutation(n)
        x, y = x[indices], y[indices]
        
        return x, y
    
    # Load the caselist data from csvs (update later to read directly from the database)
    def load_caselist(self, sensor):
        x = pd.read_csv('_data/caselist.csv')
        y = pd.read_csv('_data/sim_results.csv')

        groups = np.array(x.iloc[:, -1])
        x = np.array(x.iloc[:, 1:-1])
        y = np.array(y[sensor])
    
        return x, y, groups
    
    # Normalize the data using MinMaxScaler
    def normalize_data(self, x, y):
        scaler_x = MinMaxScaler() # Initialize the scaler for x
        scaler_y = MinMaxScaler() # Initialize the scaler for y
        
        # Reshape y to have a 2D shape if it's 1D
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        x_normalized = scaler_x.fit_transform(x) # Fit and transform the input features
        y_normalized = scaler_y.fit_transform(y) # Fit and transform the target variable
        
        # If y was originally 1D, convert it back
        if y_normalized.shape[1] == 1:
            y_normalized = y_normalized.ravel()
        if x_normalized.shape[1] == 1:
            x_normalized = x_normalized.ravel()
        
        return x_normalized, y_normalized

    # Select the initial samples using K-Medoids clustering (probably needs to be updated using another method)
    def initial_sample(self, x, y, initial_samplesize):

        # # Apply k-means clustering
        # if x.ndim == 1:
        #     kmeans = KMeans(n_clusters=min(initial_samplesize, len(x)), random_state=0).fit(x.reshape(-1, 1))
        # else:   
        #     kmeans = KMeans(n_clusters=min(initial_samplesize, len(x)), random_state=0).fit(x)
        # centroids = kmeans.cluster_centers_

        # print(centroids)

        # # Find the closest data points to the centroids of the clusters
        # indices = []
        # for center in centroids:
        #     distances = np.abs(x - center)
        #     index = np.argmin(distances)
        #     if index not in indices:
        #         indices.append(index)
        # if len(indices) < initial_samplesize: # if the distance to centroids returns the same datapoints, add 'needed' random indices
        #     extra_indices = [i for i in range(len(x)) if i not in indices]
        #     needed = initial_samplesize - len(indices)
        #     indices.extend(extra_indices[:needed])

        # print(indices)
        # print(max(indices))
        
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
    
# Based on the indices passed, transfer the data from pool to known
def update_data(k_d, p_d, indices):
    k_d_new = np.append(k_d, p_d[indices], axis=0)
    p_d_new = np.delete(p_d, indices, axis=0)
    return k_d_new, p_d_new

# Split the features and target and load the data into a torch DataLoader
def load_data(numpy_array, batch_size = 16):
    features = numpy_array[:, :-1]#.squeeze()  # All but the last column
    targets = numpy_array[:, -1]    # Last column
    print('Features and targets shape:', features.shape, targets.shape)

    torch_dataset = TensorDataset(torch.tensor(features), torch.tensor(targets))
    torch_loader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=True)
    return torch_loader