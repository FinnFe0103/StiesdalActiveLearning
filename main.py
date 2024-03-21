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
from data import Dataprep, load_data
import argparse
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter

# GP-specific imports
from Models.ExactGP import ExactGPModel
import gpytorch


class RunModel:
    def __init__(self, model_name, hidden_size, layer_number, steps, epochs, dataset_type, sensor, scaling, samples_per_step, # 0. Initialize all parameters and dataset
                 validation_size, learning_rate, active_learning, directory, verbose, run_name, complexity_weight, prior_sigma):

        # Configs
        self.run_name = run_name # Name of the run
        self.verbose = verbose # Print outputs
        os.makedirs('_plots', exist_ok=True) # Directory to save the outputs
        current_time = datetime.datetime.now().strftime("%H%M%S") # Unique directory based on datetime for each run
        log_dir = os.path.join('Models/runs', model_name, directory, run_name + '_' + current_time)
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

        # GP-specific parameters
        if self.model_name == 'GP':
            self.likelihood = None
            self.mll = None
        self.learning_rate = learning_rate # Learning rate for the optimizer

    def init_model(self, input_dim, hidden_size, layer_number, prior_sigma): # 0.1 Initialize the model
        if self.model_name == 'BNN':
            model = BayesianNetwork(input_dim, hidden_size, layer_number, prior_sigma)
        elif self.model_name == 'GP':
            model = None #model needs to be initialized with the data
    
    def init_optimizer(self, learning_rate):
        if self.model == None: # GP model needs to initialize with the data, then optimizer can be initialized
            return None 
        else:
            self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
    
    def train_model(self, step): # 1. Train the model
        # Split the pool data into train and validation sets
        if self.validation_size > 0:
            train, val = train_test_split(self.known_data, test_size=self.validation_size)

            if self.model_name == 'GP':
                # no train and val loader needed for GP, since not operating in batches
                train_loader = train
                val_loader = val
            else:
                train_loader = load_data(train)
                val_loader = load_data(val)
        elif self.validation_size == 0:
            if self.model_name == 'GP':
                # no train and val loader needed for GP, since not operating in batches
                train_loader = self.known_data
            else:
                train_loader = load_data(self.known_data)
        else:
            raise ValueError('Invalid validation size')

        if self.model_name == 'GP':
            # GP Training process
            X_train = torch.tensor(train_loader[:, :-1]).to(self.device)
            y_train = torch.tensor(train_loader[:, -1]).to(self.device)
            
            # Initialize likelihood and model with training data
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
            self.likelihood.noise = 0.01 #hyperparameter
            self.model = ExactGPModel(X_train, y_train, self.likelihood).to(self.device)
            self.init_optimizer(self.learning_rate)

            self.model.train()
            self.likelihood.train()
            self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

            for epoch in range(self.epochs):
                start_epoch = time.time()
                self.optimizer.zero_grad()
                output = self.model(X_train)
                train_loss = -self.mll(output, y_train) 
                train_loss.backward()
                self.optimizer.step()

                self.writer.add_scalar('loss/train', train_loss, step*self.epochs+epoch+1)
                if self.verbose:
                    print(f'Step: {step+1} | Epoch: {epoch+1} of {self.epochs} | Train-Loss: {train_loss:.4f} | Lengthscale: {self.model.covar_module.base_kernel.lengthscale.item():.3f} | Noise: {self.likelihood.noise.item():.3f} | {time.time() - start_epoch:.2f} seconds')
                

            if self.validation_size > 0:
                # Evaluation step here, assuming evaluate_model method exists
                self.evaluate_val_data(val_loader, step)

        else:
            # BNN Training
            self.model.to(self.device) # move the model to the configured device

            for epoch in range(self.epochs):
                start_epoch = time.time()
                self.model.train()
                total_train_loss = 0
                for x, y in train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    self.optimizer.zero_grad()

                    loss = self.model.sample_elbo(inputs=x, labels=y,
                                                criterion=self.criterion, sample_nbr=1,
                                                complexity_cost_weight=self.complexity_weight) #0.01/len(train_loader.dataset)
                    loss.backward()
                    self.optimizer.step()
                    total_train_loss += loss.item()

                # Log model parameters
                for name, param in self.model.named_parameters():
                    self.writer.add_histogram(f'{name}', param, epoch)
                    if param.grad is not None:
                        self.writer.add_histogram(f'{name}.grad', param.grad, epoch)

                # Average loss for the current epoch
                epoch_train_loss = total_train_loss / len(train_loader) # Average loss over batches
                self.writer.add_scalar('loss/train', epoch_train_loss, step*self.epochs+epoch+1)
                if self.verbose:
                    print(f'Step: {step+1} | Epoch: {epoch+1} of {self.epochs} | Train-Loss: {epoch_train_loss:.4f} | {time.time() - start_epoch:.2f} seconds')

            if self.validation_size > 0:
                self.evaluate_val_data(val_loader, step) # Evaluate the model on the validation set

    def evaluate_val_data(self, val_loader, step): # 1.1 Evaluate the model on the validation set
        if self.model_name == 'GP':
            # GP-specific evaluation process
            self.model.eval()
            self.likelihood.eval()

            X_val = torch.tensor(val_loader[:, :-1]).to(self.device)
            y_val = torch.tensor(val_loader[:, -1]).to(self.device)
            
            with torch.no_grad(), gpytorch.settings.fast_pred_var(): # There is only one "batch" for GP validation
                output = self.model(X_val)
                loss = -self.mll(output, y_val)
                total_val_loss = loss.item()

            self.writer.add_scalar('loss/val', total_val_loss, step+1)
            if self.verbose:
                print(f'Step: {step+1}, Val-Loss: {total_val_loss:.4f}')
        
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
                print(f'Step: {step+1} | Val-Loss: {step_val_loss:.4f}')
        
    def evaluate_pool_data(self, step): # 2. Evaluate the model on the pool data
        # Convert pool data to PyTorch tensors and move them to the correct device
        x_pool_torch = torch.tensor(self.pool_data[:, :-1]).to(self.device)
        y_pool_torch = torch.tensor(self.pool_data[:, -1]).unsqueeze(1).to(self.device) #.view(-1, 1).to(self.device)  # Ensure y_pool is the correct shape

        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # No need to calculate gradients
            if self.model_name == 'GP':
                # For GP, the model's output is a distribution, from which we can get the mean as predictions
                self.likelihood.eval()  # Also set the likelihood to evaluation mode
                prediction = self.model(x_pool_torch.unsqueeze(0))  # Add unsqueeze to match expected dimensions
                predictions = self.likelihood(prediction)  # Get the predictive posterior

                # Compute MSE Loss for GP
                loss = self.criterion(predictions.mean.reshape(-1, 1), y_pool_torch) 

            else:
                predictions = self.model(x_pool_torch)
                loss = self.criterion(predictions, y_pool_torch)   # Calculate the loss
            
        # Log the loss to TensorBoard
        self.writer.add_scalar('loss/pool', loss.item(), step + 1)
        
        if self.verbose:
            print(f'Step: {step + 1} | Pool-Loss: {loss.item()}')
    
    def predict(self, samples=500): # 3. Predict the mean and std (uncertainty) on the pool data
        self.model.eval()  # Set model to evaluation mode

        x_pool = self.pool_data[:, :-1]
        x_pool_torch = torch.tensor(x_pool).to(self.device)
          
        if self.model_name == 'GP':
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                self.likelihood.eval()  # Also set the likelihood to evaluation mode for GP predictions
                preds = self.model(x_pool_torch)
                observed_pred = self.likelihood(preds)
            means = observed_pred.mean.numpy()
            stds = observed_pred.stddev.numpy()
        else:
            with torch.no_grad():
                preds = [self.model(x_pool_torch) for _ in range(samples)]
            preds = torch.stack(preds)  # Shape: [samples, N, output_dim]
            means = preds.mean(dim=0).detach().cpu().numpy()  # calculate the mean of the predictions
            stds = preds.std(dim=0).detach().cpu().numpy()  # calculate the standard deviation of the predictions

        #self.preds_csv(preds, 'prediction')

        return means, stds  # Return both the mean and standard deviation of predictions

    def acquisition_function(self, means, stds, samples_per_step, step): # 4. Select the next samples from the pool
        if self.active_learning:
            uncertainty = stds.squeeze() # Uncertainty is the standard deviation of the predictions
            selected_indices = uncertainty.argsort()[-samples_per_step:] # Select the indices with the highest uncertainty
        elif self.active_learning == 'RS': # Random Sampling
            selected_indices = random.sample(range(len(self.data_pool)), samples_per_step)
        elif self.active_learning == 'EI': # Expected Improvement
            z = (means - best_y) / stds
            ei = (means - best_y) * norm.cdf(z) + stds * norm.pdf(z)
            selected_indices = ei.squeeze().argsort()[-samples_per_step:] # Select the indices with the highest EI
        elif self.active_learning == 'PI': # Probability of Improvement
            z = (means - best_y) / stds
            pi = norm.cdf(z)
            selected_indices = pi.squeeze().argsort()[-samples_per_step:] # Select the indices with the highest PI
        elif self.active_learning == 'UCB': # Upper Confidence Bound
            ucb = means + 2.0 * stds
            selected_indices = ucb.squeeze().argsort()[-samples_per_step:] # Select the indices with the highest UCB
        elif self.active_learning == 'EX': # Exploitation: next sample highest predictions
            selected_indices = np.argsort(means)[-samples_per_step:]
        else:
            selected_indices = random.sample(range(len(self.pool_data)), samples_per_step)
            return selected_indices
            
    def plot(self, means, stds, selected_indices, step, x_highest, y_highest): #4.1 Plot the predictions and selected indices
        x_pool = self.pool_data[:, :-1] # [observations, features]
        y_pool = self.pool_data[:, -1] # [observations]
        x_selected = self.known_data[:, :-1] # [observations, features]
        y_selected = self.known_data[:, -1] # [observations]
        pca_applied = False
        # Check if dimensionality reduction is needed
        if x_pool.shape[1] > 1:
            pca = PCA(n_components=1)
            x_pool = pca.fit_transform(x_pool) # [observations, 1]
            x_selected = pca.transform(x_selected) # [observations, 1]
            x_highest = pca.transform(x_highest) # [observations, 1]
            pca_applied = True
            
        x_pool_selected = x_pool[selected_indices] # [observations, 1]
        y_pool_selected = y_pool[selected_indices] # [observations]

        y_vals = [means, means + 2 * stds, means - 2 * stds] #list of 3 arrays of shape [observations in pool, 1] (use 2 for 95% CI)
        df = pd.concat([pd.DataFrame({'x': x_pool.squeeze(), 'y': y_val.squeeze()}) for y_val in y_vals], ignore_index=True)

        # Plotting
        fig = plt.figure()
        sns.lineplot(data=df, x="x", y="y")
        plt.scatter(x_pool, y_pool, c="green", marker="*", alpha=0.1)  # Plot the data pairs in the pool
        plt.scatter(x_selected, y_selected, c="red", marker="*", alpha=0.2)  # plot the train data on top
        plt.scatter(x_pool_selected, y_pool_selected, c="blue", marker="o", alpha=0.3)  # Highlight selected data points
        plt.scatter(x_highest, y_highest, c="purple", marker="o", alpha=0.3)
        plt.title(self.run_name.replace("_", " ") + f' | Step {step + 1}', fontsize='small')
        plt.xlabel('1 Principal Component' if pca_applied else 'x')
        plt.legend(['Mean prediction', 'Confidence Interval', 'Pool data (unseen)', 'Seen data', 'Selected data', 'Final Prediction'], fontsize='small')
        plt.close(fig)

        # Log the table figure
        self.writer.add_figure(f'Prediction vs Actual Table Epoch {step + 1}', fig, step + 1)
        
        if self.pool_data[:, :-1].shape[1] > 1:
            pca = PCA(n_components=2)
            data_plot = pca.fit_transform(self.pool_data[:, :-1]) # [observations, 2]
            # Plotting
            fig = plt.figure()
            plt.scatter(data_plot[:, 0], data_plot[:, 1], c='lightgray', s=30, label='Data points')
            plt.scatter(data_plot[selected_indices, 0], data_plot[selected_indices, 1], c='blue', s=50, label='Selected Samples')
            plt.title(self.run_name.replace("_", " ") + f' | Step {step + 1}', fontsize='small')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.legend(fontsize='small')
            plt.close(fig)
            
            self.writer.add_figure(f'First two PCs with selected datapoints {step + 1}', fig, step + 1)

    def update_data(self, selected_indices): # 5. Update the known and pool data
        self.known_data = np.append(self.known_data, self.pool_data[selected_indices], axis=0)
        self.pool_data = np.delete(self.pool_data, selected_indices, axis=0)

    def preds_csv(self, preds, name):
        preds = preds.detach().cpu().numpy()
        preds = preds.reshape(-1, preds.shape[-1])
        preds = pd.DataFrame(preds)
        preds.to_csv(f'{name}_preds_{self.run_name}.csv', index=False)

    def final_prediction(self, step, topk, samples=500):
        self.model.eval()
        x_total = np.concatenate((self.pool_data[:, :-1], self.known_data[:, :-1]), axis=0)
        y_total = np.concatenate((self.pool_data[:, -1], self.known_data[:, -1]), axis=0)
        
        x_total_torch = torch.tensor(x_total).to(self.device)
        
        if self.model_name == 'GP':
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                self.likelihood.eval()  # Also set the likelihood to evaluation mode for GP predictions
                preds = self.model(x_total_torch)
                observed_pred = self.likelihood(preds)
            means = observed_pred.mean.numpy()
            highest_indices = np.argsort(means)[-topk:]
        else:
            with torch.no_grad():
                preds = [self.model(x_total_torch) for _ in range(samples)]
            preds = torch.stack(preds)
            means = preds.mean(dim=0).detach().cpu().numpy().squeeze()
            highest_indices = np.argsort(means)[-topk:]

        x_highest = x_total[highest_indices]
        y_highest = y_total[highest_indices]

        return x_highest, y_highest
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run the BNN model')
    # Dataset parameters
    parser.add_argument('-ds', '--dataset_type', type=str, default='Generated_2000', help='1. Generated_2000, 2. Caselist')
    parser.add_argument('-sc', '--scaling', type=str, default='Standard', help='Scaling to be used: 1. Standard, 2. Minmax, 3. None')
    parser.add_argument('-se', '--sensor', type=str, default='foundation_origin xy FloaterOffset [m]', help='Sensor to be predicted')
    
    # Model parameters
    parser.add_argument('-m', '--model', type=str, default='BNN', help='1. BNN, 2. GP, 3. MCD, 4. DE, 5. SVM')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Learning rate')
    
    # BNN
    parser.add_argument('-hs', '--hidden_size', type=int, default=4, help='Number of hidden units')
    parser.add_argument('-ln', '--layer_number', type=int, default=3, help='Number of layers')
    parser.add_argument('-ps', '--prior_sigma', type=float, default=0.0000001, help='Prior sigma')
    parser.add_argument('-cw', '--complexity_weight', type=float, default=0.01, help='Complexity weight')

    # Active learning parameters
    parser.add_argument('-al', '--active_learning', type=str, default='UCB', help='Type of active learning/acquisition function: 1. US, 2. RS, 3. EI, 4. PI, 5. UCB') # Uncertainty, Random, Expected Improvement, Probability of Improvement, Upper Confidence Bound
    parser.add_argument('-s', '--steps', type=int, default=2, help='Number of steps')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('-ss', '--samples_per_step', type=int, default=100, help='Samples to be selected per step and initial samplesize')
    parser.add_argument('-vs', '--validation_size', type=float, default=0, help='Size of the validation set in percentage')
    parser.add_argument('-t', '--topk', type=int, default=100, help='Number of top predictions to be selected')
    
    # Output parameters    
    parser.add_argument('-dr', '--directory', type=str, default='_plots', help='Sub-directory to save the ouputs')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print the model')
    
    args = parser.parse_args()

    if args.model == 'BNN':
        opt_list = ['-sc', '-hs', '-ln', '-ps', '-cw', '-lr', '-al', '-s', '-e', '-ss', '-vs']
    elif args.model == 'GP':
        opt_list = ['-sc', '-lr', '-al', '-s', '-e', '-ss', '-vs']
    option_value_list = [(action.option_strings[0].lstrip('-'), getattr(args, action.dest)) 
                         for action in parser._actions if action.option_strings and action.option_strings[0] in opt_list]
    run_name = '_'.join(f"{abbr}{str(value)}" for abbr, value in option_value_list)
    
    model = RunModel(args.model, args.hidden_size, args.layer_number, args.steps, args.epochs, args.dataset_type, args.sensor, args.scaling, args.samples_per_step, 
                     args.validation_size, args.learning_rate, args.active_learning, args.directory, args.verbose, run_name, args.complexity_weight, args.prior_sigma)

    # Iterate through the steps of active learning
    for step in range(model.steps):
        start_step = time.time()
        model.train_model(step) # Train the model
        x_highest, y_highest = model.final_prediction(step, topk=args.topk)
        model.evaluate_pool_data(step) # Evaluate the model on the pool data
        means, stds = model.predict() # Predict the uncertainty on the pool data
        selected_indices = model.acquisition_function(means, stds, args.samples_per_step, step) # Select the next samples from the pool
        model.plot(means, stds, selected_indices, step, x_highest, y_highest) # Plot the predictions and selected indices
        model.update_data(selected_indices) # Update the known and pool data

        print(f'Updated pool and known data (AL = {args.active_learning}):', model.pool_data.shape, model.known_data.shape)
        print(f'Step: {step+1} of {model.steps} | {time.time() - start_step:.2f} seconds')

# BNN: -v -ln 3 -hs 4 -ps 0.0000001 -cw 0.01 -dr v1
# GP Demo: -v -m GP -sc Minmax -lr 0.1 -al -s 10 -e 10 -ss 25 -vs 0.2 