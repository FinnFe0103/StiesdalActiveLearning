import os
import linear_operator
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import random
import sys
import datetime
import torch
import gpytorch
import argparse
from scipy.stats import norm
from scipy.spatial.distance import cdist
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from torch.utils.tensorboard import SummaryWriter
# Module imports
from Models.BNN import BayesianNetwork
from Models.ExactGP import ExactGPModel
from Models.Ensemble import Ensemble
from Models.Dropout import Dropout

class RunModel:
    def __init__(self, model_name, run_name, directory, x_total, y_total,
                 learning_rate=None,
                 prior_sigma=None, complexity_weight=None,
                 hidden_size=None, layer_number=None,
                 ensemble_size=None, 
                 dropout_rate=None,
                 kernel=None, lengthscale_prior=None, lengthscale_sigma=None, lengthscale_mean=None, noise_prior=None, noise_sigma=None, noise_mean=None, noise_constraint=None, lengthscale_type=None,
                 acquisition_function=None, reg_lambda=None, steps=None, epochs=None,
                 verbose=False):

        # Configs
        self.run_name = run_name # Name of the run
        current_time = datetime.datetime.now().strftime("%H%M%S") # Unique directory based on datetime for each run
        self.log_dir = os.path.join(directory, model_name, current_time + '_' + run_name) # fix run_name
        self.writer = SummaryWriter(self.log_dir) # TensorBoard
        print('Run saved under:', self.log_dir)

        # Active learning parameters
        self.active_learning = acquisition_function # Which acquisition function to use
        self.reg_lambda = reg_lambda # Regularization parameter for the acquisition function
        self.steps = steps # Number of steps for the active learning
        self.epochs = epochs # Number of epochs per step of active learning
        #self.samples_per_step = samples_per_step # Number of samples to select per step
        #self.validation_size = validation_size # Size of the validation set in percentage

        # Initialize the model and optimizer
        self.learning_rate = learning_rate # Learning rate for the optimizer
        self.model_name = model_name # Name of the model
        self.hidden_size = hidden_size
        self.layer_number = layer_number
        self.prior_sigma = prior_sigma
        self.complexity_weight = complexity_weight
        self.ensemble_size = ensemble_size
        self.dropout_rate = dropout_rate
        self.init_model(x_total.shape[1]) # Initialize the model
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

        # KL Divergence
        self.set_actual_pdf(y_total) # Set the actual PDF of the data

        # Plot y data to see if data correctly
        # pca = PCA(n_components=1)
        # x_total_pca = pca.fit_transform(x_total) # [observations, 1]
        # plt.scatter(x_total_pca, y_total)
        # plt.show()

    def init_model(self, input_dim): # 0.1 Initialize the model
        if self.model_name == 'BNN':
            self.complexity_weight = self.complexity_weight # Complexity weight for ELBO
            self.model = BayesianNetwork(input_dim, self.hidden_size, self.layer_number, self.prior_sigma)
            self.init_optimizer_criterion() # Initialize the optimizer
        elif self.model_name == 'GP': # Model and optimizer are initialized in the training step
            self.model = None
            self.likelihood = None # Initialize the likelihood
            self.mll = None # Initialize the marginal log likelihood
        elif self.model_name == 'SVR':
            self.model = SVR(kernel='rbf', C=5, epsilon=0.05)
            self.init_optimizer_criterion() # Initialize the optimizer
        elif self.model_name == 'DE':
            self.model = [Ensemble(input_dim, self.hidden_size, self.layer_number) for _ in range(self.ensemble_size)]
            self.init_optimizer_criterion() # Initialize the optimizer
        elif self.model_name == 'MCD':
            self.model = Dropout(input_dim, self.hidden_size, self.layer_number, self.dropout_rate)
            self.init_optimizer_criterion()
    
    def init_optimizer_criterion(self): # 0.2 Initialize the optimizer
        if self.model_name == 'BNN':
            self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
            self.mse = torch.nn.MSELoss() # MSE
            self.mae = torch.nn.L1Loss() # MAE
        elif self.model_name == 'GP':
            self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
            self.mse = torch.nn.MSELoss() # MSE
            self.mae = torch.nn.L1Loss() # MAE
        elif self.model_name == 'SVR':
            self.optimizer = None # SVR does not have an optimizer
            self.mse = mean_squared_error # MSE
            self.mae = mean_absolute_error # MAE
        elif self.model_name == 'DE':
            self.optimizer = [Adam(model.parameters(), lr=self.learning_rate) for model in self.model]
            self.mse = torch.nn.MSELoss() # MSE
            self.mae = torch.nn.L1Loss() # MAE
        elif self.model_name == 'MCD':
            self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
            self.mse = torch.nn.MSELoss() # MSE
            self.mae = torch.nn.L1Loss() # MAE

    def train_model(self, step, X_selected, y_selected): # 1. Train the model
        if self.model_name == 'BNN': # BNN Training
            self.model.to(self.device) # Move the model to the configured device

            torch_dataset = TensorDataset(torch.tensor(X_selected), torch.tensor(y_selected).unsqueeze(1))
            train_loader = DataLoader(torch_dataset, batch_size=16, shuffle=True)

            for epoch in range(self.epochs):
                start_epoch = time.time()
                self.model.train() # Set the model to training mode
                total_train_loss = 0

                for x, y in train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    self.optimizer.zero_grad()
                    loss = self.model.sample_elbo(inputs=x, labels=y,
                                                criterion=self.mse, sample_nbr=1,
                                                complexity_cost_weight=self.complexity_weight) #0.01/len(train_loader.dataset)
                    loss.backward()
                    self.optimizer.step()
                    total_train_loss += loss.item()

                # Average loss for the current epoch
                epoch_train_loss = total_train_loss / len(train_loader) # Average loss over batches
                self.writer.add_scalar('loss/train', epoch_train_loss, step*self.epochs+epoch+1)
                #print(f'Step: {step+1} | Epoch: {epoch+1} of {self.epochs} | Train-Loss: {epoch_train_loss:.4f} | {time.time() - start_epoch:.2f} seconds')

        elif self.model_name == 'GP': # GP Training
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

        elif self.model_name == 'SVR': # SVR Training
            self.model.fit(X_selected, y_selected)

        elif self.model_name == 'DE': # DE Training

            torch_dataset = TensorDataset(torch.tensor(X_selected), torch.tensor(y_selected).unsqueeze(1))
            train_loader = DataLoader(torch_dataset, batch_size=16, shuffle=True)

            model_epoch_loss = []
            for model_idx in range(len(self.model)):
                self.model[model_idx].to(self.device) # Move the model to the configured device
                
                epoch_loss = []
                for epoch in range(self.epochs):    
                    self.model[model_idx].train() # Set the model to training mode
                    total_train_loss = 0

                    for x, y in train_loader:
                        x, y = x.to(self.device), y.to(self.device)
                        self.optimizer[model_idx].zero_grad()
                        loss = self.mse(self.model[model_idx](x), y.squeeze())
                        loss.backward()
                        self.optimizer[model_idx].step()
                        total_train_loss += loss.item()
                    epoch_loss.append(total_train_loss / len(train_loader))
                model_epoch_loss.append(epoch_loss)
            train_loss = sum(sublist[-1] for sublist in model_epoch_loss) / len(model_epoch_loss)
            self.writer.add_scalar('loss/train', train_loss, step+1)
            #print(f'Step: {step+1} | Train-Loss: {train_loss:.4f}')

        elif self.model_name == 'MCD': # MCD Training

            torch_dataset = TensorDataset(torch.tensor(X_selected), torch.tensor(y_selected).unsqueeze(1))
            train_loader = DataLoader(torch_dataset, batch_size=16, shuffle=True)

            self.model.to(self.device)
            for epoch in range(self.epochs):
                start_epoch = time.time()
                self.model.train()
                total_train_loss = 0
                for x, y in train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    self.optimizer.zero_grad()
                    loss = self.mse(self.model(x), y.squeeze())
                    loss.backward()
                    self.optimizer.step()
                    total_train_loss += loss.item()

                # Average loss for the current epoch
                epoch_train_loss = total_train_loss / len(train_loader) # Average loss over batches
                self.writer.add_scalar('loss/train', epoch_train_loss, step*self.epochs+epoch+1)
                #print(f'Step: {step+1} | Epoch: {epoch+1} of {self.epochs} | Train-Loss: {epoch_train_loss:.4f} | {time.time() - start_epoch:.2f} seconds')

    def final_prediction(self, step, X_total, y_total, X_selected, topk): # 2. Final prediction on the total data
        X_total_torch = torch.tensor(X_total).to(self.device)
        y_total_torch = torch.tensor(y_total).to(self.device)  # y pool torch: [observations, ]
        
        if self.model_name == 'BNN':
            self.model.eval()
            with torch.no_grad():
                preds = [self.model(X_total_torch) for _ in range(100)]
            preds = torch.stack(preds) # [100, observations, 1]
            means = preds.mean(dim=0).detach().cpu().numpy().squeeze() # [observations, ]
            mse = self.mse(torch.tensor(means).to(self.device), y_total_torch)
            mae = self.mae(torch.tensor(means).to(self.device), y_total_torch)
        elif self.model_name == 'GP':
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
        elif self.model_name == 'SVR':
            means = self.model.predict(X_total)
            mse = self.mse(means, y_total)
            mae = self.mae(means, y_total)
        elif self.model_name == 'DE':
            for model in self.model:
                model.eval()
            with torch.no_grad():
                preds = [model(X_total_torch).clone().detach().cpu().numpy() for model in self.model]
            means = np.mean(preds, axis=0)
            mse = self.mse(torch.tensor(means).to(self.device), y_total_torch.squeeze())
            mae = self.mae(torch.tensor(means).to(self.device), y_total_torch.squeeze())
        elif self.model_name == 'MCD':
            #self.model.eval() # DO NOT PUT IN EVALUATION MODE; OTHERWISE DROPOUT WILL NOT WORK
            with torch.no_grad():
                preds = [self.model(X_total_torch) for _ in range(100)]
            preds = torch.stack(preds)
            means = preds.mean(dim=0).detach().cpu().numpy()
            mse = self.mse(torch.tensor(means).to(self.device), y_total_torch.squeeze())
            mae = self.mae(torch.tensor(means).to(self.device), y_total_torch.squeeze())
            
        highest_indices_pred_n = np.argsort(means)[-topk:]
        highest_indices_pred_1 = np.argsort(means)[-1]
        highest_indices_actual_n = np.argsort(y_total)[-topk:]
        highest_indices_actual_1 = np.argsort(y_total)[-1]

        x_highest_pred_n = X_total[highest_indices_pred_n]
        y_highest_pred_n = y_total[highest_indices_pred_n]

        x_highest_actual_n = X_total[highest_indices_actual_n]
        y_highest_actual_n = y_total[highest_indices_actual_n]

        x_highest_actual_1 = X_total[highest_indices_actual_1]
        y_highest_actual_1 = y_total[highest_indices_actual_1]

        common_indices = np.intersect1d(highest_indices_pred_n, highest_indices_actual_n)
        percentage_common = (len(common_indices) / len(highest_indices_pred_n)) * 100

        preds_from_known = 0
        for row1 in x_highest_pred_n:
            for row2 in X_selected:
                if np.array_equal(row1, row2):
                    preds_from_known += 1
                    break

        # Log the loss to TensorBoard
        self.writer.add_scalar('loss/total_mse', mse.item(), step + 1)
        self.writer.add_scalar('loss/total_mae', mae.item(), step + 1)

        highest_actual_in_top = False
        if any(np.array_equal(row, x_highest_actual_1) for row in x_highest_pred_n):
            highest_actual_in_top = True
            print("FOUND: The highest actual value is in the top predictions")
        else:
            print("NOT FOUND: The highest actual value is not in the top predictions")
        
        print(f'Percentage of common indices in top {topk} predictions: {percentage_common:.2f}%')
        print(f'Number of predictions from pool: {len(x_highest_pred_n)-preds_from_known} | Number of predictions from known data: {preds_from_known}')

        highest_actual_in_known = False
        if any(np.array_equal(row, x_highest_actual_1) for row in X_selected):
            highest_actual_in_known = True
            print("KNOWN: The highest actual value is in the known data")
        else:
            print("NOT KNOWN: The highest actual value is not in the known data")
        
        self.set_predicted_pdf(means)
        
        return x_highest_pred_n, y_highest_pred_n, x_highest_actual_n, y_highest_actual_n, x_highest_actual_1, y_highest_actual_1, mse.item(), mae.item(), percentage_common, highest_actual_in_top, highest_actual_in_known

    def evaluate_pool_data(self, step, X_pool, y_pool): # 3. Evaluate the model on the pool data
        # Convert pool data to PyTorch tensors and move them to the correct device
        X_pool_torch = torch.tensor(X_pool).to(self.device)
        y_pool_torch = torch.tensor(y_pool).to(self.device) # [observations, ]

        if self.model_name == 'BNN':
            self.model.eval()  # Set the model to evaluation mode
            with torch.no_grad():  # No need to calculate gradients
                predictions = self.model(X_pool_torch).squeeze() # [observations, ]
                mse = self.mse(predictions, y_pool_torch)
                mae = self.mae(predictions, y_pool_torch)
                
        elif self.model_name == 'GP':
            self.model.eval()  # Set the model to evaluation mode
            with torch.no_grad():  # No need to calculate gradients
                self.likelihood.eval()  # Also set the likelihood to evaluation mode
                prediction = self.model(X_pool_torch) # [observations, ]
                predictions = self.likelihood(prediction)  # [observations, ]
                mse = self.mse(predictions.mean, y_pool_torch) 
                mae = self.mae(predictions.mean, y_pool_torch)

        elif self.model_name == 'SVR':
            predictions = self.model.predict(X_pool)
            mse = self.mse(predictions, y_pool)
            mae = self.mae(predictions, y_pool)

        elif self.model_name == 'DE':
            for model in self.model:
                model.eval()
            with torch.no_grad():
                predictions = [model(X_pool_torch).clone().detach().cpu().numpy() for model in self.model]
                predictions = np.mean(predictions, axis=0)
                mse = self.mse(torch.tensor(predictions).to(self.device), y_pool_torch.squeeze())
                mae = self.mae(torch.tensor(predictions).to(self.device), y_pool_torch.squeeze())

        elif self.model_name == 'MCD':
            #self.model.eval()  # DO NOT PUT IN EVALUATION MODE; OTHERWISE DROPOUT WILL NOT WORK
            with torch.no_grad():
                predictions = self.model(X_pool_torch)
                mse = self.mse(predictions, y_pool_torch.squeeze())
                mae = self.mae(predictions, y_pool_torch.squeeze())
            
        # Log the loss to TensorBoard
        self.writer.add_scalar('loss/pool_mse', mse.item(), step + 1)
        self.writer.add_scalar('loss/pool_mae', mae.item(), step + 1)
        print(f'Step: {step + 1} | Pool-Loss: {mse.item()}')

    def predict(self, X_pool): # 4. Predict the mean and std (uncertainty) on the pool data
        X_pool_torch = torch.tensor(X_pool).to(self.device)

        if self.model_name == 'BNN':
            self.model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                preds = [self.model(X_pool_torch) for _ in range(500)]
            preds = torch.stack(preds)  # Shape: [samples, N, output_dim]
            means = preds.mean(dim=0).squeeze().detach().cpu().numpy()  # calculate the mean of the predictions
            stds = preds.std(dim=0).squeeze().detach().cpu().numpy()  # calculate the standard deviation of the predictions
        elif self.model_name == 'GP':
            self.model.eval()  # Set model to evaluation mode
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                self.likelihood.eval()  # Also set the likelihood to evaluation mode for GP predictions
                preds = self.model(X_pool_torch)
                observed_pred = self.likelihood(preds)
            means = observed_pred.mean.numpy()
            stds = observed_pred.stddev.numpy()
        elif self.model_name == 'SVR':
            means = self.model.predict(X_pool)
            stds = np.zeros_like(means)
        elif self.model_name == 'DE':
            for model in self.model:
                model.eval()
            with torch.no_grad():
                preds = [model(X_pool_torch).clone().detach().cpu().numpy() for model in self.model]
                means = np.mean(preds, axis=0)
                stds = np.std(preds, axis=0)
        elif self.model_name == 'MCD':
            #self.model.eval() # DO NOT PUT IN EVALUATION MODE; OTHERWISE DROPOUT WILL NOT WORK
            with torch.no_grad():
                preds = [self.model(X_pool_torch) for _ in range(500)]
            preds = torch.stack(preds)
            means = preds.mean(dim=0).detach().cpu().numpy()
            stds = preds.std(dim=0).detach().cpu().numpy()


            ##########################################

            # # Plotting
            # fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
            # for idx in range(len(preds)):
            #     print(x_pool_torch.cpu().shape, preds[idx].shape)
            #     ax.scatter(x_pool_torch.cpu().squeeze(), preds[idx])
            # ax.scatter(self.data_pool[:, :-1], self.data_pool[:, -1], c='red', marker='*', alpha=0.1)
            # plt.show()

            ##########################################

        return means, stds  # Return both the mean and standard deviation of predictions

    def acquisition_function(self, means, stds, y_selected, X_Pool, topk, selected_indices): # 5. Select the next samples from the pool
        stds = stds.copy()
        means = means.copy()

        best_y = np.max(y_selected)
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
                elif self.active_learning == 'EI':
                    z = (means - best_y) / stds
                    scores = (means - best_y) * norm.cdf(z) + stds * norm.pdf(z) + self.reg_lambda * current_min_distances
                elif self.active_learning == 'PI':
                    z = (means - best_y) / stds
                    scores = norm.cdf(z) + self.reg_lambda * current_min_distances
                elif self.active_learning == 'UCB':
                    scores = means + 2.0 * stds + self.reg_lambda * current_min_distances

                indices_sorted_by_score = np.argsort(scores)[::-1]  # Sort indices by scores in descending order
                for index in indices_sorted_by_score:
                    if index not in selected_indices:  # Ensure the index wasn't already selected
                        selected_indices.append(index)  # Update the list of selected indices
                        break

        return selected_indices

    def plot(self, means, stds, selected_indices, step, x_highest_pred_n, y_highest_pred_n, x_highest_actual_n, y_highest_actual_n, x_highest_actual_1, y_highest_actual_1, X_pool, y_pool, X_selected, y_selected): #6 Plot the predictions and selected indices # , kl_divergence_for_plot, x_points, p, q, y_min_extended, y_max_extended
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
        print(x_pool_selected.shape, y_pool_selected.shape)
        print(selected_indices)

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

        # Log the table figure
        # self.writer.add_figure(f'Prediction vs Actual Table Epoch {step + 1}', fig, step + 1)

        
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
            
            #self.writer.add_figure(f'First two PCs with selected datapoints {step + 1}', fig, step + 1)

        # PDFs and KL divergence
        # fig, axs = plt.subplots(1, 2, figsize=(18, 6))

        # axs[0].plot(x_points, p, color='blue', label='Actual')
        # axs[0].plot(x_points, q, color='red', label='Predicted')
        # axs[0].legend(loc='upper right')
        # axs[0].set_title('PDFs of Actual and Predicted Data')
        # axs[0].set_xlabel('Data Points (Predicted and Actual Output)')
        # axs[0].set_ylabel('(Log-) Probability Density')

        # Plot the KL divergence
        # axs[1].plot(x_points, kl_divergence_for_plot, color='purple')
        # axs[1].fill_between(x_points, kl_divergence_for_plot, color='purple', alpha=0.1)
        # axs[1].set_title('Pointwise KL Divergence')
        # axs[1].set_xlabel('Data Points (Predicted and Actual Output)')
        # axs[1].set_ylabel('Pointwise KL Divergence')

        # plt.tight_layout()
        # plt.savefig(f'{self.log_dir}/PDFs and KL Divergence {step + 1}.png')
        # plt.close(fig)
        #self.writer.add_figure(f'PDFs and KL Divergence {step + 1}', fig, step + 1)


    def set_actual_pdf(self, y_total):
        self.pdf_p = self.calc_kde_pdf(y_total) # Calculate the actual PDF of the data
    
    def set_predicted_pdf(self, y_total_predicted):
        self.pdf_q = self.calc_kde_pdf(y_total_predicted) # Calculate the predicted PDF of the data
    
    def get_kl_divergence(self):
        """
        Calculate the KL divergence between two PDFs.

        pdf_p: Callable PDF of the first distribution (e.g., actual y-values).
        pdf_q: Callable PDF of the second distribution (e.g., predicted y-values).
        x_points: Points at which to evaluate the PDFs and calculate the divergence.
        """
        # Assuming y_total_actual and y_total_predicted are already defined
        y_min = np.min(self.y_total_actual)
        y_max = np.max(self.y_total_actual)

        # Extend the range slightly to ensure coverage of the tails
        extend_factor = 0.1  # Extend by 10% of the range on both sides
        range_extend = (y_max - y_min) * extend_factor
        y_min_extended = y_min - range_extend
        y_max_extended = y_max + range_extend

        # Generate x_points
        x_points = np.linspace(y_min_extended, y_max_extended, 1000)

        # Points at which to evaluate the PDFs
        p = self.pdf_p(x_points)
        q = self.pdf_q(x_points)
        # Ensure q is nonzero to avoid division by zero in log
        q = np.maximum(q, np.finfo(float).eps)
        
        # Calculate the KL divergence
        kl_divergence_for_plot = p * np.log(p / q) 
        kl_divergence = np.sum(p * np.log(p / q)) * (x_points[1] - x_points[0])

        return kl_divergence_for_plot, x_points, p, q, y_min_extended, y_max_extended, kl_divergence
    
    def calc_kde_pdf(self, y_data):
        y_reshaped = y_data.reshape(-1, 1)  # Reshape data for KDE

        # Extend the grid search to include multiple kernels and a broader range of bandwidths
        params = {
            'bandwidth': np.logspace(-2, 1, 30),  # Broader and more fine-grained range for bandwidth
            'kernel': ['gaussian'] 
        }
        grid = GridSearchCV(KernelDensity(), params, cv=5)
        grid.fit(y_reshaped)

        # Use the best estimator to compute the KDE
        kde = grid.best_estimator_

        # Creating a PDF function based on KDE for multivariate data
        def kde_pdf(x):
            x_reshaped = np.atleast_2d(x).reshape(-1, 1)  # Ensure x is 2D for score_samples
            log_dens = kde.score_samples(x_reshaped)
            return np.exp(log_dens)
        #print(f"Best bandwidth: {grid.best_estimator_.bandwidth}, Best kernel: {grid.best_estimator_.kernel}")

        return kde_pdf

    def create_single_kernel(self, kernel_type, input_dim):
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