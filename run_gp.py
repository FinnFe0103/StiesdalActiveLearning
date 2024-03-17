import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import random
import datetime
import torch
from Models.BNN import BayesianNetwork
from Models.ExactGP import ExactGPModel
from data import Dataprep, load_data
import argparse
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.preprocessing import MinMaxScaler

class RunModel:
    def __init__(self, model_name, hidden_size, layer_number, steps, epochs, dataset_type, sensor, scaling, samples_per_step, # 0. Initialize all parameters and dataset
                 validation_size, learning_rate, active_learning, directory, verbose, run_name, complexity_weight, prior_sigma):

        # Configs
        self.run_name = run_name # Name of the run
        self.verbose = verbose # Print outputs
        os.makedirs(directory, exist_ok=True) # Directory to save the outputs
        current_time = datetime.datetime.now().strftime("%m_%d-%H_%M_%S") # Unique directory based on datetime for each run
        log_dir = os.path.join('Models/runs', model_name, run_name) + '_' + current_time
        self.writer = SummaryWriter(log_dir) # TensorBoard
        print('Run saved under:', log_dir)

        # Data parameters
        self.data = Dataprep(dataset_type, sensor, scaling=scaling, initial_samplesize=samples_per_step)
        self.known_data, self.pool_data = self.data.known_data, self.data.pool_data

        # Active learning parameters
        self.active_learning = active_learning # Whether to use active learning or random sampling
        self.steps = steps # Number of steps for the active learning
        self.epochs = epochs # Number of epochs per step of active learning
        self.validation_size = validation_size # Size of the validation set in percentage

        # Initialize the model and optimizer
        self.model_name = model_name # Name of the model
        self.model = self.init_model(self.known_data.shape[1]-1, hidden_size, layer_number, prior_sigma) # Initialize the model
        self.criterion = torch.nn.MSELoss() # Loss function
        self.optimizer = self.init_optimizer(learning_rate) # Initialize the optimizer
        self.device = torch.device("mps" if torch.backends.mps.is_available() and self.model_name != "GP" else "cpu")
        print(f'Using {self.device} for training')
        self.complexity_weight = complexity_weight # Complexity weight for ELBO
        self.learning_rate = learning_rate # Learning rate for the optimizer

        # GP-specific parameters
        if self.model_name == 'GP':
            self.likelihood = None
            self.mll = None

    def init_model(self, input_dim, hidden_size, layer_number, prior_sigma): # 0.1 Initialize the model
        if self.model_name == 'BNN':
            model = BayesianNetwork(input_dim, hidden_size, layer_number, prior_sigma)
        elif self.model_name == 'GP':
            model = None

        #     model = MCDropout(hidden_size)
        # elif self.model_name == 'Deep Ensembles':
        #     model = DeepEnsembles(hidden_size)
        else:
            raise ValueError('Invalid model type')
        return model
    
    def init_optimizer(self, learning_rate):
        if self.model == None: 
            return None
        else:
            optimizer = Adam(self.model.parameters(), lr=learning_rate)
        return optimizer
    
    def train_model(self, step):
        # Split the pool data into train and validation sets
        if self.validation_size > 0:
            train, val = train_test_split(self.known_data, test_size=self.validation_size)
            if self.model_name == 'GP':
                train_loader = train
                val_loader = val
            else:
                train_loader = load_data(train)
                val_loader = load_data(val)
        elif self.validation_size == 0:
            if self.model_name == 'GP':
                train_loader = self.known_data
            else:
                train_loader = load_data(self.known_data)
        else:
            raise ValueError('Invalid validation size')

        if self.model_name == 'GP':
            # GP Training process
            #scaler = MinMaxScaler()
            #X_train = torch.tensor(scaler.fit_transform(train_loader[:, 0].reshape(-1, 1))).to(self.device)  # Adds an extra dimension to make X_train a 2D tensor
            #y_train = torch.tensor(train_loader[:, 1]).to(self.device)

            X_train = torch.tensor(train_loader[:, :-1]).to(self.device)
            y_train = torch.tensor(train_loader[:, -1]).to(self.device)
            
            # Initialize likelihood and model with training data
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
            self.likelihood.noise = 0.01
            self.model = ExactGPModel(X_train, y_train, self.likelihood).to(self.device)
            self.optimizer = self.init_optimizer(self.learning_rate)

            self.model.train()
            self.likelihood.train()
            self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

            for epoch in range(self.epochs):
                start_epoch = time.time()
                self.optimizer.zero_grad()
                output = self.model(X_train)
                loss = -self.mll(output, y_train) 
                loss.backward()
                if self.verbose:
                    print(f'Step: {step+1} | Epoch: {epoch+1} of {self.epochs} | Train-Loss: {loss:.4f} | Lengthscale: {self.model.covar_module.base_kernel.lengthscale.item():.3f} | Noise: {self.likelihood.noise.item():.3f} | {time.time() - start_epoch:.2f} seconds')
                self.optimizer.step()
            if self.validation_size > 0:
                # Evaluation step here, assuming evaluate_model method exists
                self.evaluate_val_data(val_loader, step)

        else:
            # BNN Training process
            self.model.to(self.device)  # Move the model to the configured device

            for epoch in range(self.epochs):
                start_epoch = time.time()
                self.model.train()
                total_train_loss = 0
                for x, y in train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    self.optimizer.zero_grad()
                    loss = self.model.sample_elbo(inputs=x, labels=y,
                                                  criterion=self.criterion, sample_nbr=1,
                                                  complexity_cost_weight=self.complexity_weight)
                    loss.backward()
                    self.optimizer.step()
                    total_train_loss += loss.item()

                # Logging and printing
                epoch_train_loss = total_train_loss / len(train_loader)
                if self.verbose:
                    print(f'Step: {step+1} | Epoch: {epoch+1} of {self.epochs} | Train-Loss: {epoch_train_loss:.4f} | {time.time() - start_epoch:.2f} seconds')

            if self.validation_size > 0:
                self.evaluate_val_data(val, val_loader, step)  # Evaluate on the validation set

    def evaluate_val_data(self, val_loader, step):
         # 1.1 Evaluate the model on the validation set
        if self.model_name == 'GP':
            # GP-specific evaluation process
            self.model.eval()
            self.likelihood.eval()

            X_val = torch.tensor(val_loader[:, :-1]).to(self.device)
            y_val = torch.tensor(val_loader[:, -1]).to(self.device)
            
            total_val_loss = 0
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                output = self.model(X_val)
                loss = -self.mll(output, y_val)
                total_val_loss += loss.item()

            step_val_loss = total_val_loss  # There is only one "batch" for GP validation
            self.writer.add_scalar('loss/val', step_val_loss, step+1)
            if self.verbose:
                print(f'Step: {step+1}, Val-Loss: {step_val_loss:.4f}')

        else: 
            self.model.eval()  # Set the model to evaluation mode
            total_val_loss = 0

            with torch.no_grad():  # Inference mode, gradients not required
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    # Calculate loss using sample_elbo for Bayesian inference
                    loss = self.model.sample_elbo(inputs=x, labels=y,
                                                criterion=self.criterion, sample_nbr=3,
                                                complexity_cost_weight=self.complexity_weight)
                    total_val_loss += loss.item()

            step_val_loss = total_val_loss / len(val_loader) # Average loss over batches
            self.writer.add_scalar('loss/val', step_val_loss, step+1)
            if self.verbose:
                print('Step: {} | Val-Loss: {:.4f}'.format(step+1, step_val_loss))
        
    def evaluate_pool_data(self, step):
        # Convert pool data to PyTorch tensors and move them to the correct device
        x_pool_torch = torch.tensor(self.pool_data[:, :-1]).to(self.device)
        y_pool_torch = torch.tensor(self.pool_data[:, -1]).to(self.device)
        
        #x_pool_torch = torch.tensor(self.pool_data[:, :-1], dtype=torch.float32).to(self.device)
        #y_pool_torch = torch.tensor(self.pool_data[:, -1], dtype=torch.float32).unsqueeze(1).to(self.device)

        self.model.eval()  # Set the model to evaluation mode
        
        with torch.no_grad():  # No need to calculate gradients
            if self.model_name == 'GP':
                # For GP, the model's output is a distribution, from which we can get the mean as predictions
                self.likelihood.eval()  # Also set the likelihood to evaluation mode
                print(x_pool_torch.shape)
                prediction = self.model(x_pool_torch.unsqueeze(0))  # Add unsqueeze to match expected dimensions
                predictions = self.likelihood(prediction)  # Get the predictive posterior
                
                # Compute MSE Loss for GP
                mse_loss = torch.mean((predictions.mean - y_pool_torch) ** 2)
                loss = mse_loss  # Or use a different GP-specific criterion, if preferred
            else:
                predictions = self.model(x_pool_torch)
                loss = self.criterion(predictions, y_pool_torch)  # Calculate the loss using self.criterion, assuming it's MSE for non-GP models
            
        # Log the loss to TensorBoard
        self.writer.add_scalar('loss/pool', loss.item(), step + 1)
        
        if self.verbose:
            print(f'Step: {step + 1} | Pool-Loss: {loss.item()}')
    
    def predict(self):
        self.model.eval()  # Set model to evaluation mode

        x_pool = self.pool_data[:, :-1]
        x_pool_torch = torch.tensor(x_pool, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            if self.model_name == 'GP':
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    self.likelihood.eval()  # Also set the likelihood to evaluation mode for GP predictions
                    preds = self.model(x_pool_torch)
                    observed_pred = self.likelihood(preds)
                    means = observed_pred.mean.numpy()
                    stds = observed_pred.stddev.numpy()
            else:
                # Assuming self.model(x) returns a distribution from which you can sample for non-GP models
                preds = [self.model(x_pool_torch) for _ in range(500)]
                preds = torch.stack(preds)  # Shape: [samples, N, output_dim]
                means = preds.mean(dim=0).detach().cpu().numpy()  # calculate the mean of the predictions
                stds = preds.std(dim=0).detach().cpu().numpy()  # calculate the standard deviation of the predictions

        # Optionally, save predictions to CSV
        # self.preds_csv(preds, 'prediction')
        return means, stds  # Return both the mean and standard deviation of predictions

    def acquisition_function(self, means, stds, samples_per_step, step): # 4. Select the next samples from the pool
        if self.active_learning:
            uncertainty = stds.squeeze() # Uncertainty is the standard deviation of the predictions
            selected_indices = uncertainty.argsort()[-samples_per_step:] # Select the indices with the highest uncertainty
            return selected_indices
            
        else:
            selected_indices = random.sample(range(len(self.pool_data)), samples_per_step)
            return selected_indices
            
    def plot(self, means, stds, selected_indices, step):
        # Assuming self.model_name exists and distinguishes between GP and other models
        x_pool = self.pool_data[:, :-1]  # [observations, features]
        y_pool = self.pool_data[:, -1]  # [observations]
        x_selected = self.known_data[:, :-1]  # [observations, features]
        y_selected = self.known_data[:, -1]  # [observations]
        pca_applied = False

        # Check if dimensionality reduction is needed
        if x_pool.shape[1] > 1:
            pca = PCA(n_components=1 if self.model_name == 'GP' else 2)
            x_pool = pca.fit_transform(x_pool)  # [observations, 1 or 2]
            x_selected = pca.transform(x_selected)  # [observations, 1 or 2]
            pca_applied = True

        x_pool_selected = x_pool[selected_indices]  # [observations, 1 or 2]
        y_pool_selected = y_pool[selected_indices]  # [observations]

        y_vals = [means, means + 2 * stds, means - 2 * stds]  # list of 3 arrays of shape [observations in pool, 1] (use 2 for 95% CI)
        df = pd.concat([pd.DataFrame({'x': x_pool.squeeze(), 'y': y_val.squeeze()}) for y_val in y_vals], ignore_index=True)

        # Plotting
        fig = plt.figure()
        sns.lineplot(data=df, x="x", y="y")
        plt.scatter(x_pool, y_pool, c="green", marker="*", alpha=0.1)  # Plot the data pairs in the pool
        plt.scatter(x_selected, y_selected, c="red", marker="*", alpha=0.2)  # plot the train data on top
        plt.scatter(x_pool_selected, y_pool_selected, c="blue", marker="o", alpha=0.3)  # Highlight selected data points
        plt.title(self.run_name.replace("_", " ") + f' | Step {step + 1}', fontsize='small')
        plt.xlabel('1 Principal Component' if pca_applied else 'x')
        plt.legend(['Mean prediction', 'Confidence Interval', 'Pool data (unseen)', 'Seen data', 'Selected data'], fontsize='small')
        plt.close(fig)

        # Log the figure
        self.writer.add_figure(f'Prediction vs Actual Table Epoch {step + 1}', fig, step + 1)

        # Additional 2D PCA plot for non-GP models or if needed
        if pca_applied and self.model_name != 'GP':
            # This block can be adjusted or extended for GP-specific visualization if needed
            pass

    def update_data(self, selected_indices): # 5. Update the known and pool data
        self.known_data = np.append(self.known_data, self.pool_data[selected_indices], axis=0)
        self.pool_data = np.delete(self.pool_data, selected_indices, axis=0)

    def preds_csv(self, preds, name):
        preds = preds.detach().cpu().numpy()
        preds = preds.reshape(-1, preds.shape[-1])
        preds = pd.DataFrame(preds)
        preds.to_csv(f'{name}_preds_{self.run_name}.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the BNN model')
    # Dataset parameters
    parser.add_argument('-ds', '--dataset_type', type=str, default='Generated_2000', help='1. Generated_2000, 2. Caselist')
    parser.add_argument('-sc', '--scaling', type=str, default='Minmax', help='Scaling to be used: 1. Standard, 2. Minmax, 3. None')
    parser.add_argument('-se', '--sensor', type=str, default='foundation_origin xy FloaterOffset [m]', help='Sensor to be predicted')
    
    # Model parameters
    parser.add_argument('-m', '--model', type=str, default='BNN', help='1. BNN, 2. GP, 3. Deep Ensembles')
    parser.add_argument('-hs', '--hidden_size', type=int, default=10, help='Number of hidden units')
    parser.add_argument('-ln', '--layer_number', type=int, default=10, help='Number of layers')
    parser.add_argument('-ps', '--prior_sigma', type=float, default=0.1, help='Prior sigma')
    parser.add_argument('-cw', '--complexity_weight', type=float, default=0.1, help='Complexity weight')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1, help='Learning rate')
    
    # Active learning parameters
    parser.add_argument('-al', '--active_learning', action='store_true', help='Use active learning (AL) or random sampling (RS)')
    parser.add_argument('-s', '--steps', type=int, default=50, help='Number of steps')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('-ss', '--samples_per_step', type=int, default=20, help='Samples to be selected per step and initial samplesize')
    parser.add_argument('-vs', '--validation_size', type=float, default=0.2, help='Size of the validation set in percentage')
    
    # Output parameters    
    parser.add_argument('-dr', '--directory', type=str, default='_plots', help='Directory to save the ouputs')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print the model')
    parser.add_argument('-rn', '--run_name', type=str, default='hello', help='Run name prefix')
    
    args = parser.parse_args()

    outside = ['-h', '-ds', '-m', '-v', '-rn', '-dr', '-vs', '-se'] 
    option_value_list = [(action.option_strings[0].lstrip('-'), getattr(args, action.dest)) 
                         for action in parser._actions if action.option_strings and action.option_strings[0] not in outside]
    run_name = args.run_name + '_' + '_'.join(f"{abbr}{str(value)}" for abbr, value in option_value_list)
    
    model = RunModel(args.model, args.hidden_size, args.layer_number, args.steps, args.epochs, args.dataset_type, args.sensor, args.scaling, args.samples_per_step, 
                     args.validation_size, args.learning_rate, args.active_learning, args.directory, args.verbose, run_name, args.complexity_weight, args.prior_sigma)
    #-v -ln 3 -hs 4 -ps 0.0000001 -cw 0.01 -rn v1
    # Iterate through the steps of active learning
    for step in range(model.steps):
        start_step = time.time()
        model.train_model(step) # Train the model
        model.evaluate_pool_data(step) # Evaluate the model on the pool data
        means, stds = model.predict() # Predict the uncertainty on the pool data
        selected_indices = model.acquisition_function(means, stds, args.samples_per_step, step) # Select the next samples from the pool
        model.plot(means, stds, selected_indices, step) # Plot the predictions and selected indices
        model.update_data(selected_indices) # Update the known and pool data

        print(f'Updated pool and known data (AL = {args.active_learning}):', model.pool_data.shape, model.known_data.shape)
        print('Step: {} of {} | {:.2f} seconds'.format(step+1, model.steps, time.time() - start_step))




    