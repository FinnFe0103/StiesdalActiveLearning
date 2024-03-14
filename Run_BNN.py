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

class RunModel:
    def __init__(self, model_name, hidden_size, layer_number, steps, epochs, dataset_type, sensor, scaling, samples_per_step, 
                 validation_size, learning_rate, active_learning, directory, verbose, run_name):

        # Configs
        self.run_name = run_name # Name of the run
        self.verbose = verbose # Print outputs
        os.makedirs(directory, exist_ok=True) # Directory to save the outputs
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # Unique directory based on datetime for each run
        log_dir = os.path.join('Models/runs', run_name + '_' + current_time)
        self.writer = SummaryWriter(log_dir) # TensorBoard
        print('Run saved under:', log_dir)

        self.data = Dataprep(dataset_type, sensor, scaling=scaling, initial_samplesize=samples_per_step)
        self.known_data, self.pool_data = self.data.known_data, self.data.pool_data

        self.validation_size = validation_size # Size of the validation set in percentage

        # Active learning parameters
        self.active_learning = active_learning # Whether to use active learning or random sampling
        self.steps = steps # Number of steps for the active learning
        self.epochs = epochs # Number of epochs per step of active learning

        # Initialize the model and optimizer
        self.model_name = model_name # Name of the model
        input_dim = self.known_data.shape[1]-1 # Input dimension 
        self.model = self.init_model(input_dim, hidden_size, layer_number) # Initialize the model
        self.criterion = torch.nn.MSELoss() # Loss function
        self.optimizer = self.init_optimizer(learning_rate) # Initialize the optimizer
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print('Using device:', self.device)

    def init_model(self, input_dim, hidden_size, layer_number):
        if self.model_name == 'BNN':
            model = BayesianNetwork(input_dim, hidden_size, layer_number)
        # elif self.model_name == 'MC Dropout':
        #     model = MCDropout(hidden_size)
        # elif self.model_name == 'Deep Ensembles':
        #     model = DeepEnsembles(hidden_size)
        else:
            raise ValueError('Invalid model type')
        return model
    
    def init_optimizer(self, learning_rate):
        optimizer = Adam(self.model.parameters(), lr=learning_rate)
        return optimizer
    
    def train_model(self, step):
        # Split the pool data into train and validation sets
        if self.validation_size > 0:
            train, val = train_test_split(self.known_data, test_size=self.validation_size)
            train_loader = load_data(train)
            val_loader = load_data(val)
        elif self.validation_size == 0:
            train_loader = load_data(self.known_data)
            print("No validation set")
        else:
            raise ValueError('Invalid validation size')

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
                                              complexity_cost_weight=0) #0.01/len(train_loader.dataset)
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
                print('Step: {} | Epoch: {} of {} | Train-Loss: {:.4f} | {:.2f} seconds'.format(step+1, epoch+1, self.epochs, epoch_train_loss, time.time() - start_epoch))

        if self.validation_size > 0:
            self.evaluate_val_data(val_loader, step) # Evaluate the model on the validation set

    def evaluate_val_data(self, val_loader, step): # 1: EVALUATE THE MODEL ON THE VALIDATION SET
        self.model.eval()  # Set the model to evaluation mode
        total_val_loss = 0

        with torch.no_grad():  # Inference mode, gradients not required
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                # Calculate loss using sample_elbo for Bayesian inference
                loss = self.model.sample_elbo(inputs=x, labels=y,
                                            criterion=self.criterion, sample_nbr=3,
                                            complexity_cost_weight=0) #0.01/len(val_loader.dataset)
                total_val_loss += loss.item()

        step_val_loss = total_val_loss / len(val_loader) # Average loss over batches
        self.writer.add_scalar('loss/val', step_val_loss, step+1)
        if self.verbose:
            print('Step: {} | Val-Loss: {:.4f}'.format(step+1, step_val_loss))
    
    def predict(self, samples=1000): # 2: PREDICT THE UNCERTAINTY ON THE POOL DATA
        self.model.eval()  # Set model to evaluation mode

        x_pool = self.pool_data[:, :-1]
        x_pool_torch = torch.tensor(x_pool).to(self.device)
        with torch.no_grad():
            preds = [self.model(x_pool_torch) for _ in range(samples)]
        preds = torch.stack(preds)  # Shape: [samples, N, output_dim]
        means = preds.mean(dim=0).detach().cpu().numpy()  # calculate the mean of the predictions
        stds = preds.std(dim=0).detach().cpu().numpy()  # calculate the standard deviation of the predictions

        return means, stds  # Return both the mean and standard deviation of predictions
    
    def evaluate_pool_data(self, step):
        # Convert pool data to PyTorch tensors and move them to the correct device
        x_pool_torch = torch.tensor(self.pool_data[:, :-1]).to(self.device)
        y_pool_torch = torch.tensor(self.pool_data[:, -1]).unsqueeze(1).to(self.device) #.view(-1, 1).to(self.device)  # Ensure y_pool is the correct shape

        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # No need to calculate gradients
            predictions = self.model(x_pool_torch)
            loss = self.criterion(predictions, y_pool_torch)  # Calculate the loss
            
        # Log the loss to TensorBoard
        self.writer.add_scalar('loss/pool', loss.item(), step + 1)
        
        if self.verbose:
            print(f'Step: {step + 1} | Pool-Loss: {loss.item()}')

    def acquisition_function(self, means, stds, samples_per_step, step): # 3: DECIDE WHICH SAMPLES TO SELECT FROM THE POOL DATA (USING ACQUISITION FUNCTION)
        if self.active_learning:
            uncertainty = stds.squeeze() # Uncertainty is the standard deviation of the predictions
            selected_indices = uncertainty.argsort()[-samples_per_step:] # Select the indices with the highest uncertainty
            return selected_indices
            if self.known_data.shape[1] == 2:
                self.plot(means, stds, selected_indices, step) # Plot the predictions and selected indices
            
        else:
            selected_indices = random.sample(range(len(self.pool_data)), samples_per_step)
            return selected_indices
            if self.known_data.shape[1] == 2:
                self.plot(means, stds, selected_indices, step) # Plot the predictions and selected indices
            
    def update_data(self, selected_indices): # Based on the indices passed, transfer the data from pool to known
        self.known_data = np.append(self.known_data, self.pool_data[selected_indices], axis=0)
        self.pool_data = np.delete(self.pool_data, selected_indices, axis=0)

    def plot(self, means, stds, selected_indices, step):
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
            pca_applied = True
            
        x_pool_selected = x_pool[selected_indices] # [observations, 1]
        y_pool_selected = y_pool[selected_indices] # [observations]

        y_vals = [means, means + 100 * stds, means - 100 * stds] #list of 3 arrays of shape [observations in pool, 1] (use 2 for 95% CI)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the BNN model')
    parser.add_argument('-m', '--model', type=str, default='BNN', help='1. BNN, 2. MC Dropout, 3. Deep Ensembles')
    parser.add_argument('-hs', '--hidden_size', type=int, default=4, help='Number of hidden units')
    parser.add_argument('-ln', '--layer_number', type=int, default=3, help='Number of layers')
    parser.add_argument('-s', '--steps', type=int, default=2, help='Number of steps')
    parser.add_argument('-e', '--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('-ds', '--dataset_type', type=str, default='Generated_2000', help='1. Generated_2000, 2. Caselist')
    parser.add_argument('-se', '--sensor', type=str, default='foundation_origin xy FloaterOffset [m]', help='Sensor to be predicted')
    parser.add_argument('-sc', '--scaling', type=str, default='none', help='Scaling to be used: 1. standard, 2. minmax, 3. none')
    parser.add_argument('-is', '--samples_per_step', type=int, default=100, help='Samples to be selected per step and initial samplesize')
    parser.add_argument('-vs', '--validation_size', type=float, default=0, help='Size of the validation set in percentage')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('-al', '--active_learning', action='store_true', help='Use active learning (AL) or random sampling (RS)')
    parser.add_argument('-dr', '--directory', type=str, default='_plots', help='Directory to save the ouputs')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print the model')
    args = parser.parse_args()

    run_name = args.model + '_' + args.dataset_type + '_scaling=' + args.scaling + '_AL=' + str(args.active_learning) + '_Steps=' + str(args.steps) + '_Epochs=' + str(args.epochs)

    model = RunModel(args.model, args.hidden_size, args.layer_number, args.steps, args.epochs, args.dataset_type, args.sensor, args.scaling, args.samples_per_step, 
                     args.validation_size, args.learning_rate, args.active_learning, args.directory, args.verbose, run_name)

    # Iterate through the steps of active learning
    for step in range(model.steps):
        start_step = time.time()
        model.train_model(step) # Train the model
        model.evaluate_pool_data(step) # Evaluate the model on the pool data
        means, stds = model.predict() # Predict the uncertainty on the pool data
        selected_indices = model.acquisition_function(means, stds, args.samples_per_step, step) # Select the next samples from the pool
        #if args.dataset_type == 'Generated_2000':
        model.plot(means, stds, selected_indices, step) # Plot the predictions and selected indices
        model.update_data(selected_indices) # Update the known and pool data

        print(f'After updating the pool and known data using AL = {args.active_learning}:', model.pool_data.shape, model.known_data.shape)
        print('Step: {} of {} | {:.2f} seconds'.format(step+1, model.steps, time.time() - start_step))