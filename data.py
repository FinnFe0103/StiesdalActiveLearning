import numpy as np
import math
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset, DataLoader
import torch


class Dataprep:
    def __init__(self, dataset_type, initial_samplesize, sensor):

        if dataset_type == 'Caselist':
            self.x, self.y, self.groups = self.load_caselist(sensor)
        elif dataset_type.split('_')[0] == 'Generated':
            dataset_size = int(dataset_type.split('_')[1])
            self.x, self.y = self.generate_data(-10, 10, dataset_size)

        self.known_data, self.pool_data = self.initial_sample(self.x, self.y, initial_samplesize)

    def generate_data(self, start, end, n):
        x = np.linspace(start, end, n)
        sample_mean = [math.sin(i/2) for i in x]
        sample_var = [((abs(start)+abs(end))/2 - abs(i))/16 for i in x]
        y = stats.norm(sample_mean, sample_var).rvs()
        
        indices = np.random.permutation(n) # Shuffle the x and y values
        x, y = x[indices], y[indices]
        
        return x, y
    
    def load_caselist(self, sensor):
        x = pd.read_csv('_data/caselist.csv')
        y = pd.read_csv('_data/sim_results.csv')

        groups = np.array(x.iloc[:, -1])
        x = np.array(x.iloc[:, 1:-1])
        y = np.array(y[sensor])
    
        return x, y, groups

    def initial_sample(self, x, y, initial_samplesize):
        print(len(x))
        # Apply k-means clustering
        if x.ndim == 1:
            kmeans = KMeans(n_clusters=min(initial_samplesize, len(x)), random_state=0).fit(x.reshape(-1, 1))
        else:   
            kmeans = KMeans(n_clusters=min(initial_samplesize, len(x)), random_state=0).fit(x)
        centroids = kmeans.cluster_centers_

        # Find the closest data points to the centroids of the clusters
        indices = []
        for center in centroids:
            distances = np.abs(x - center)
            index = np.argmin(distances)
            if index not in indices:
                indices.append(index)
        if len(indices) < initial_samplesize: # if the distance to centroids returns the same datapoints, add 'needed' random indices
            extra_indices = [i for i in range(len(x)) if i not in indices]
            needed = initial_samplesize - len(indices)
            indices.extend(extra_indices[:needed])

        print(indices)
        print(max(indices))

        # Create the modified tensors based on the selected indices
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

        print(indices)

        return known_data, pool_data
    
    def next_sample(self, indices):
        #change = self.pool_data[indices]

        self.known_data = np.append(self.known_data, self.pool_data[indices], axis=0)
        self.pool_data = np.delete(self.pool_data, indices, axis=0)
        
        # print(indices)
        
        # change_df = pd.DataFrame(change)
        # pool_df = pd.DataFrame(self.pool_data)
        # selected_df = pd.DataFrame(self.known_data)
        # return pool_df, selected_df, change_df

        return self.known_data, self.pool_data # WHY DOES IT NOT RETURN THE SELF VALUES???
    

    def load_data(self, numpy_array, batch_size = 16):
        features = numpy_array[:, :-1]  # All but the last column
        targets = numpy_array[:, -1]    # Last column
        torch_dataset = TensorDataset(torch.tensor(features), torch.tensor(targets))
        torch_loader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=True)

        return torch_loader