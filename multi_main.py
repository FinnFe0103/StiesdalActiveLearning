import os
import linear_operator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import random
import sys
import datetime
import torch
import gpytorch
from tqdm import tqdm
from scipy.stats import norm
from scipy.spatial.distance import cdist
from torch.optim import Adam
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter
# Module imports
from Models.ExactGP import ExactGPModel

class RunModel:
    def __init__(self, model_name, run_name, directory,
                 learning_rate=None,
                 kernel=None, lengthscale_prior=None, lengthscale_sigma=None, lengthscale_mean=None, noise_prior=None, noise_sigma=None, noise_mean=None, noise_constraint=None, lengthscale_type=None,
                 acquisition_function=None, reg_lambda=None, steps=None, epochs=None):

        # Configs
        self.run_name = run_name # Name of the run
        current_time = datetime.datetime.now().strftime("%H%M%S") # Unique directory based on datetime for each run
        self.log_dir = os.path.join(directory, model_name, current_time + '_' + run_name) # fix run_name
        self.writer = SummaryWriter(self.log_dir) # TensorBoard

        # Active learning parameters
        self.active_learning = acquisition_function # Which acquisition function to use
        self.reg_lambda = reg_lambda # Regularization parameter for the acquisition function
        self.steps = steps # Number of steps for the active learning
        self.epochs = epochs # Number of epochs per step of active learning

        # Initialize the model and optimizer
        self.learning_rate = learning_rate # Learning rate for the optimizer
        self.model_name = model_name # Name of the model
        
        self.init_model() # Initialize the model
        self.device = torch.device('mps' if torch.backends.mps.is_available() and self.model_name != 'GP' else 'cpu') #and self.model_name != 'DE' 
        
        # GP Parameters
        self.kernel_type = kernel # Kernel type for GP
        self.lengthscale_prior = lengthscale_prior
        self.noise_prior = noise_prior
        self.lengthscale_sigma = lengthscale_sigma # Lengthscale for GP
        self.noise_sigma = noise_sigma # Noise for GP
        self.lengthscale_mean = lengthscale_mean
        self.noise_mean = noise_mean
        self.noise_constraint = noise_constraint
        self.lengthscale_type = lengthscale_type

    def init_model(self): # 0.1 Initialize the model
        '''
        Initialize the model, likelihood, optimizer, and cost function
        '''

        self.model = None
        self.likelihood = None # Initialize the likelihood
        self.mll = None # Initialize the marginal log likelihood
    
    def init_optimizer_criterion(self): # 0.2 Initialize the optimizer
        '''
        Initialize the optimizer and the loss function
        '''

        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.mse = torch.nn.MSELoss() # MSE
        self.mae = torch.nn.L1Loss() # MAE

    def train_model(self, step, X_selected, y_selected): # 1. Train the model
        '''
        Train the model on the selected data
        '''

        X_train = torch.tensor(X_selected).to(self.device)
        y_train = torch.tensor(y_selected).to(self.device)

        # Noise & Likelihood
        noise_prior, noise_constraint, noise_sigma = self.init_noise(self.noise_prior, self.noise_sigma, self.noise_mean)
        if noise_prior is not None:
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=noise_constraint)
            self.likelihood.noise_covar.register_prior("noise_prior", noise_prior, "noise")
        elif noise_sigma is not None:
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
            self.likelihood.noise = noise_sigma
        else:
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        # Kernel
        self.kernel = self.init_kernel(self.kernel_type, X_train.shape[1])
        # Model
        self.model = ExactGPModel(X_train, y_train, self.likelihood, self.kernel)
        # Optimizer
        self.init_optimizer_criterion()
        # Cost function
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        # Set the model and likelihood to training mode
        self.model.train()
        self.likelihood.train()

        try:
            for epoch in range(self.epochs):
                start_epoch = time.time()
                self.optimizer.zero_grad()
                output = self.model(X_train)
                train_loss = -self.mll(output, y_train) 
                train_loss.backward()
                self.optimizer.step()

                self.writer.add_scalar('loss/train', train_loss, step*self.epochs+epoch+1)
                #print(f'Step: {step+1} | Epoch: {epoch+1} of {self.epochs} | Train-Loss: {train_loss:.4f}  | Noise: {self.likelihood.noise.item():.3f} | {time.time() - start_epoch:.2f} seconds')
        except linear_operator.utils.errors.NotPSDError:
            print("Warning: The matrix is not positive semi-definite. Exiting this run.")
            sys.exit()

    def final_prediction(self, step, X_total, y_total, X_selected, topk): # 2. Final prediction on the total data
        '''
        Get the final predictions on the total data

        Parameters:
        - step: The current step
        - X_total: The total data
        - y_total: The total data
        - X_selected: The selected data
        - topk: The top k predictions to consider

        Returns:
        - x_highest_pred_n: The top n predictions
        - y_highest_pred_n: The top n predictions
        - x_highest_actual_n: The top n actual values
        - y_highest_actual_n: The top n actual values
        - x_highest_actual_1: The highest actual value
        - y_highest_actual_1: The highest actual value
        - mse: Mean Squared Error
        - mae: Mean Absolute Error
        - percentage_common: Percentage of common values in the top n predictions and actual values
        - index_of_actual_1_in_pred: Index of the highest actual value in the predictions
        - seen_count: Number of times the highest actual value was seen in the predictions
        - highest_actual_in_top: Whether the highest actual value was in the top predictions
        - highest_indices_pred: Indices of the highest predictions
        - highest_indices_actual_1: Index of the highest actual value
        '''

        X_total_torch = torch.tensor(X_total).to(self.device)
        y_total_torch = torch.tensor(y_total).to(self.device)  # y pool torch: [observations, ]
        
        self.model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            self.likelihood.eval()  # Also set the likelihood to evaluation mode for GP predictions
            try:
                pred = self.model(X_total_torch)
                preds = self.likelihood(pred)
                means = preds.mean.numpy()
                mse = self.mse(preds.mean, y_total_torch) # y total torch: [observations, ]
                mae = self.mae(preds.mean, y_total_torch) # y total torch: [observations, ]
            except linear_operator.utils.errors.NotPSDError:
                print("Warning: The matrix is not positive semi-definite. Exiting this run.")
                sys.exit()

        highest_indices_pred = np.argsort(means)[::-1]
        highest_indices_pred_n = highest_indices_pred[:topk]

        highest_indices_actual = np.argsort(y_total)[::-1]
        highest_indices_actual_n = highest_indices_actual[:topk]
        highest_indices_actual_1 = np.argsort(y_total)[-1]

        index_of_actual_1_in_pred = np.where(highest_indices_pred == highest_indices_actual_1)[0][0]

        top_n_pred_indices = highest_indices_pred[:index_of_actual_1_in_pred+1]

        seen_count = 0
        for idx in top_n_pred_indices:
            if any(np.array_equal(X_total[idx], x) for x in X_selected):
                seen_count += 1

        # For plotting
        x_highest_pred_n = X_total[highest_indices_pred_n]
        y_highest_pred_n = y_total[highest_indices_pred_n]

        x_highest_actual_n = X_total[highest_indices_actual_n]
        y_highest_actual_n = y_total[highest_indices_actual_n]

        x_highest_actual_1 = X_total[highest_indices_actual_1]
        y_highest_actual_1 = y_total[highest_indices_actual_1]

        common_indices = np.intersect1d(highest_indices_pred_n, highest_indices_actual_n)
        percentage_common = (len(common_indices) / len(highest_indices_pred_n)) * 100

        # Log the loss to TensorBoard
        self.writer.add_scalar('loss/total_mse', mse.item(), step + 1)
        self.writer.add_scalar('loss/total_mae', mae.item(), step + 1)

        highest_actual_in_top = False
        if any(np.array_equal(row, x_highest_actual_1) for row in x_highest_pred_n):
            highest_actual_in_top = True
            tqdm.write("FOUND: The highest actual value is in the top predictions")
        else:
            tqdm.write("NOT FOUND: The highest actual value is not in the top predictions")
        
        return x_highest_pred_n, y_highest_pred_n, x_highest_actual_n, y_highest_actual_n, x_highest_actual_1, y_highest_actual_1, mse.item(), mae.item(), percentage_common, index_of_actual_1_in_pred, seen_count, highest_actual_in_top, highest_indices_pred, highest_indices_actual_1

    def predict(self, X_pool): # 3. Predict the mean and std (uncertainty) on the pool data
        '''
        Predict the mean and standard deviation on the pool data
        
        Parameters:
        - X_pool: The pool data
        
        Returns:
        - means: The mean of the predictions
        - stds: The standard deviation of the predictions
        '''

        X_pool_torch = torch.tensor(X_pool).to(self.device)

        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            self.likelihood.eval()  # Also set the likelihood to evaluation mode for GP predictions
            preds = self.model(X_pool_torch)
            observed_pred = self.likelihood(preds)
        means = observed_pred.mean.numpy()
        stds = observed_pred.stddev.numpy()

        return means, stds  # Return both the mean and standard deviation of predictions

    def acquisition_function(self, means, stds, y_selected, X_Pool, topk, selected_indices): # 4. Select the next samples from the pool
        '''
        Select the next samples from the pool
        
        Parameters:
        - means: The mean of the predictions
        - stds: The standard deviation of the predictions
        - y_selected: The selected data
        - X_Pool: The pool data
        - topk: The number of samples to select
        - selected_indices: The indices of the selected data so far
        
        Returns:
        - selected_indices: The updated indices of the selected data
        '''

        stds = stds.copy()
        means = means.copy()

        distance_matrix = cdist(X_Pool, X_Pool, 'euclidean') # Distance matrix between data points
        initial_lenght = len(selected_indices)

        # Process based on the active learning strategy
        if self.active_learning == 'RS':
            while len(selected_indices) < initial_lenght+topk:
                index = random.sample(range(len(X_Pool)), 1)[0]
                if index not in selected_indices:
                    selected_indices.append(index)
        elif self.active_learning == 'EX':
            sorted_indices = np.argsort(means)[::-1]
            for idx in sorted_indices:
                if idx not in selected_indices:
                    selected_indices.append(idx)
                    if len(selected_indices) == initial_lenght+topk:
                        break
        else:
            while len(selected_indices) < initial_lenght+topk:
                if selected_indices:
                    current_min_distances = distance_matrix[:, selected_indices].min(axis=1)
                else:
                    current_min_distances = np.zeros(len(X_Pool))

                if self.active_learning == 'US':
                    scores = stds + self.reg_lambda * current_min_distances    
                elif self.active_learning == 'UCB':
                    scores = means + 2.0 * stds + self.reg_lambda * current_min_distances

                indices_sorted_by_score = np.argsort(scores)[::-1]  # Sort indices by scores in descending order
                for index in indices_sorted_by_score:
                    if index not in selected_indices:  # Ensure the index wasn't already selected
                        selected_indices.append(index)  # Update the list of selected indices
                        break

        return selected_indices

    def plot(self, means, stds, selected_indices, step, x_highest_pred_n, y_highest_pred_n, x_highest_actual_n, y_highest_actual_n, x_highest_actual_1, y_highest_actual_1, X_pool, y_pool, X_selected, y_selected): #5. Plot the predictions and selected indices # , kl_divergence_for_plot, x_points, p, q, y_min_extended, y_max_extended
        '''
        Plot the predictions and selected indices
        
        Parameters:
        - means: The mean of the predictions
        - stds: The standard deviation of the predictions
        - selected_indices: The indices of the selected data
        - step: The current step
        - x_highest_pred_n: The top n predictions
        - y_highest_pred_n: The top n predictions
        - x_highest_actual_n: The top n actual values
        - y_highest_actual_n: The top n actual values
        - x_highest_actual_1: The highest actual value
        - y_highest_actual_1: The highest actual value
        - X_pool: The pool data
        - y_pool: The pool data
        - X_selected: The selected data
        - y_selected: The selected data
        
        Returns:
        - None (saves the plot to the log directory)
        '''

        pca_applied = False
        # Check if dimensionality reduction is needed
        if X_pool.shape[1] > 1:
            pca = PCA(n_components=1)
            X_pool = pca.fit_transform(X_pool) # [observations, 1]
            X_selected = pca.transform(X_selected) # [observations, 1]
            x_highest_pred_n = pca.transform(x_highest_pred_n) # [observations, 1]
            x_highest_actual_n = pca.transform(x_highest_actual_n) # [observations, 1]
            x_highest_actual_1 = pca.transform(x_highest_actual_1.reshape(1, -1)) # [observations, 1]
            #print('Explained variance by the first princiapal components:', pca.explained_variance_ratio_)
            pca_applied = True
            
        x_pool_selected = X_pool[selected_indices] # [observations, 1]
        y_pool_selected = y_pool[selected_indices] # [observations]

        y_vals = [means, means + 2 * stds, means - 2 * stds] #list of 3 arrays of shape [observations in pool, 1] (use 2 for 95% CI)
        df = pd.concat([pd.DataFrame({'x': X_pool.squeeze(), 'y': y_val.squeeze()}) for y_val in y_vals], ignore_index=True)

        # Plotting
        fig = plt.figure(figsize=(8, 6), dpi=120)
        sns.lineplot(data=df, x="x", y="y", alpha=0.2)
        plt.scatter(X_pool, y_pool, c="green", marker="*", alpha=0.1)  # Plot the data pairs in the pool
        plt.scatter(X_selected, y_selected, c="red", marker="*", alpha=0.1)  # plot the train data on top
        plt.scatter(x_pool_selected, y_pool_selected, c="blue", marker="o", alpha=0.8)  # Highlight selected data points
        plt.scatter(x_highest_pred_n, y_highest_pred_n, c="purple", marker="o", alpha=0.8)
        plt.scatter(x_highest_actual_n, y_highest_actual_n, c="orange", marker="o", alpha=0.1)
        plt.scatter(x_highest_actual_1, y_highest_actual_1, c="red", marker="o", alpha=0.1)
        plt.title(self.run_name.replace("_", " ") + f' | Step {step + 1}', fontsize=6)
        plt.xlabel('1 Principal Component' if pca_applied else 'x')
        plt.legend(['Mean prediction', 'Confidence Interval', 'Pool data (unseen)', 'Seen data', 'Selected data', 'Final Prediction', 'Highest Actual'], fontsize=8)
        plt.savefig(f'{self.log_dir}/Prediction vs Actual Table Epoch {step + 1}.png')
        plt.close(fig)
        
        if X_pool.shape[1] > 1:
            pca = PCA(n_components=2)
            data_plot = pca.fit_transform(X_pool) # [observations, 2]
            # Plotting
            fig = plt.figure(figsize=(8, 6), dpi=120)
            plt.scatter(data_plot[:, 0], data_plot[:, 1], c='lightgray', s=30, label='Data points')
            plt.scatter(data_plot[selected_indices, 0], data_plot[selected_indices, 1], c='blue', s=50, label='Selected Samples')
            plt.title(self.run_name.replace("_", " ") + f' | Step {step + 1}', fontsize=10)
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.legend(fontsize='small')
            plt.savefig(f'{self.log_dir}/First two PCs with selected datapoints {step + 1}.png')
            plt.close(fig)




    ###########################################
    #### Helper functions for the GP model ####
    ###########################################

    def create_single_kernel(self, kernel_type, input_dim):
        '''
        Create a single kernel based on the type
        
        Parameters:
        - kernel_type: The type of kernel to create
        - input_dim: The input dimensionality of the kernel
        
        Returns:
        - kernel: The initialized kernel
        '''

        # If ARD is True, the kernel has a different lengthscale for each input dimension
        if self.lengthscale_type == 'ARD':
            ard_num_dims = input_dim
        else:
            ard_num_dims = None

        # Function to initialize a single kernel based on the type
        if kernel_type == 'RBF':
            kernel = gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
        elif kernel_type == 'Matern':
            kernel = gpytorch.kernels.MaternKernel(ard_num_dims=ard_num_dims)
        elif kernel_type == 'Linear':
            kernel = gpytorch.kernels.LinearKernel(ard_num_dims=ard_num_dims)
        elif kernel_type == 'Cosine':
            kernel = gpytorch.kernels.CosineKernel(ard_num_dims=ard_num_dims)
        elif kernel_type == 'Periodic':
            kernel = gpytorch.kernels.PeriodicKernel()
        elif kernel_type == 'RationalQuadratic':
            kernel = gpytorch.kernels.RQKernel(ard_num_dims=ard_num_dims)
        else:
            raise ValueError(f'Invalid kernel type: {kernel_type}')

        # Apply lengthscale prior if specified
        if self.lengthscale_prior is not None and hasattr(kernel, 'lengthscale'):
            if lengthscale_prior == 'Normal':
                lengthscale_prior = gpytorch.priors.NormalPrior(self.lengthscale_mean, self.lengthscale_sigma)
            elif lengthscale_prior == 'Gamma':
                lengthscale_prior = gpytorch.priors.GammaPrior(self.lengthscale_mean, self.lengthscale_sigma)
                kernel.lengthscale_constraint = gpytorch.constraints.Positive()
            else:
                raise ValueError('Invalid prior type')

            kernel.register_prior("lengthscale_prior", lengthscale_prior, "lengthscale")
            
        # Apply lengthscale if specified
        elif self.lengthscale_sigma is not None and not hasattr(kernel, 'lengthscale'):
            kernel.lengthscale = self.lengthscale_sigma
        
        return kernel

    def init_kernel(self, kernel_types, input_dim):
        '''
        Initialize the kernel based on the types provided
        
        Parameters:
        - kernel_types: The types of kernels to combine
        - input_dim: The input dimensionality of the kernel
        
        Returns:
        - combined_kernel: The (combined) kernel
        '''

        # Initialize an empty list to hold the individual kernels
        kernels = []

        # Loop through the list of kernel types provided and initialize each
        for kernel_type in kernel_types.split('+'):  # Assume kernel_types is a string like "RBF+Linear" or "RBF*Periodic"
            if '*' in kernel_type:  # Handle multiplicative combinations
                subkernels = kernel_type.split('*')
                kernel_product = None
                for subkernel_type in subkernels:
                    subkernel = self.create_single_kernel(subkernel_type.strip(), input_dim)
                    kernel_product = subkernel if kernel_product is None else kernel_product * subkernel
                kernels.append(kernel_product)
            else:
                kernels.append(self.create_single_kernel(kernel_type.strip(), input_dim))
        
        # Combine the kernels
        combined_kernel = None
        for kernel in kernels:
            combined_kernel = kernel if combined_kernel is None else combined_kernel + kernel

        return combined_kernel
    
    def init_noise(self, noise_prior, noise_sigma=None, noise_mean=None, noise_constraint=1e-6):
        '''
        Initialize the noise prior
        
        Parameters:
        - noise_prior: The noise prior to use
        - noise_sigma: The noise sigma
        - noise_mean: The noise mean
        - noise_constraint: The noise constraint
        
        Returns:
        - noise_prior: The noise prior
        - constraint: The noise constraint
        - noise_sigma: The noise sigma
        '''

        if noise_prior is not None and noise_sigma is not None and noise_mean is not None:
            if noise_prior == 'Normal':
                constraint = gpytorch.constraints.GreaterThan(noise_constraint)
                noise_prior = gpytorch.priors.NormalPrior(noise_mean, noise_sigma)
            elif noise_prior == 'Gamma':
                constraint = gpytorch.constraints.GreaterThan(noise_constraint)
                noise_prior = gpytorch.priors.GammaPrior(noise_mean, noise_sigma)
            else:
                raise ValueError('Invalid noise prior')
    
            return noise_prior, constraint, noise_sigma
    
        elif noise_sigma is not None:
            noise_constraint = None
            return None, None, noise_sigma
        else:
            return None, None, None