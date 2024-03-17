import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from Models.ExactGP import ExactGPModel
from data import Dataprep, update_data, load_data
import argparse
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random
import datetime
import torch

class RunModel:
    def __init__(self, model_name, hidden_size, steps, epochs, dataset_type, sensor, samples_per_step, 
                 validation_size, learning_rate, active_learning, directory, tensorboard, verbose):
        
        # Configs
        self.verbose = verbose # Print outputs
        os.makedirs(directory, exist_ok=True) # Directory to save the outputs
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # Unique directory based on datetime for each run
        log_dir = os.path.join('Models/runs', 'BNN_' + dataset_type  + '_' + current_time)
        self.writer = SummaryWriter(log_dir) # TensorBoard

        self.data = Dataprep(dataset_type, sensor, initial_samplesize=samples_per_step)
        self.known_data, self.pool_data = self.data.known_data, self.data.pool_data

        self.validation_size = validation_size # Size of the validation set in percentage

        # Active learning parameters
        self.active_learning = active_learning # Whether to use active learning or random sampling
        self.steps = steps # Number of steps for the active learning
        self.epochs = epochs # Number of epochs per step of active learning

        # Initialize the model and optimizer
        self.model_name = model_name # Name of the model
        input_dim = self.known_data.shape[1]-1 # Input dimension 
        self.model = self.init_model(input_dim, hidden_size) # Initialize the model
        self.criterion = torch.nn.MSELoss() # Loss function
        self.optimizer = self.init_optimizer(learning_rate) # Initialize the optimizer
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(self.device)

    def init_model(self, input_dim, hidden_size):
        if self.model_name == 'BNN':
            model = BayesianNetwork(input_dim, hidden_size)
        elif self.model_name == 'ExactGP':
            # Split the pool data into train and validation sets
            train, val = train_test_split(self.known_data, test_size=self.validation_size)
            train_loader = load_data(train)
            val_loader = load_data(val)
            model = ExactGPModel()
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
        print('K_D, P_D', self.known_data.shape, self.pool_data.shape)

        

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
            self.writer.add_scalar('loss/train', epoch_train_loss, step*10+epoch+1)
            if self.verbose:
                print('Step: {}, Epoch: {} of {}, Train-Loss: {:.4f}, time-taken: {:.2f} seconds'.format(step+1, epoch+1, self.epochs, epoch_train_loss, time.time() - start_epoch))

        self.evaluate_model(val_loader, step) # Evaluate the model on the validation set

    def evaluate_model(self, val_loader, step): # 1: EVALUATE THE MODEL ON THE VALIDATION SET
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
            print('Step: {}, Val-Loss: {:.4f}'.format(step+1, step_val_loss))

        if self.known_data.shape[1] == 2:
            self.plot(step)
    
    def plot(self, step, samples=500):
        x_pool = self.pool_data[:, :-1]
        y_pool = self.pool_data[:, -1]

        x_selected = self.known_data[:, :-1]
        y_selected = self.known_data[:, -1]

        x_pool_torch = torch.tensor(x_pool) 
        preds = [self.model(x_pool_torch.to(self.device)) for _ in range(samples)]
        preds = torch.stack(preds)  # Shape: [samples, N, output_dim]
        means = preds.mean(dim=0).detach().cpu().numpy()  # calculate the mean of the predictions
        stds = preds.std(dim=0).detach().cpu().numpy()  # calculate the standard deviation of the predictions

        # Prepare data for plotting
        dfs = []
        y_vals = [means, means + 2 * stds, means - 2 * stds]

        for i in range(3): #len(y_vals)
            dfs.append(pd.DataFrame({'x': x_pool.squeeze(), 'y': y_vals[i].squeeze()}))
        df = pd.concat(dfs)

        # Plotting
        fig = plt.figure()
        sns.lineplot(data=df, x="x", y="y")
        plt.scatter(x_pool, y_pool, c="green", marker="*", alpha=0.1)  # Plot actual y values
        plt.scatter(x_selected, y_selected, c="red", marker="*", alpha=0.2) # plot train data on top
        plt.title(f'Predictions vs Actual Step {step+1}')
        plt.legend(['Mean prediction', 'Pool data (unseen)', 'Seen data'], fontsize='small')
        plt.close(fig)

        # Log the table figure
        self.writer.add_figure(f'Prediction vs Actual Table Epoch {step+1}', fig, step+1)

    def predict(self, x): # 2: PREDICT THE UNCERTAINTY ON THE POOL DATA
        pass

    def acquisition_function(self, n): # 3: DECIDE WHICH SAMPLES TO SELECT FROM THE POOL DATA (USING ACQUISITION FUNCTION)
        selected_indices = random.sample(range(len(self.pool_data)), n)
        self.known_data, self.pool_data = update_data(self.known_data, self.pool_data, selected_indices)

        print(self.pool_data.shape)
        print(self.known_data.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the BNN model')
    parser.add_argument('-m', '--model', type=str, default='ExactGP', help='1. BNN, 2. MC Dropout, 3. Deep Ensembles')
    parser.add_argument('-hs', '--hidden_size', type=int, default=4, help='Number of hidden units')
    parser.add_argument('-s', '--steps', type=int, default=2, help='Number of steps')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('-ds', '--dataset_type', type=str, default='Generated_2000', help='1. Generated_2000, 2. Caselist')
    parser.add_argument('-se', '--sensor', type=str, default='foundation_origin xy FloaterOffset [m]', help='Sensor to be predicted')
    parser.add_argument('-is', '--samples_per_step', type=int, default=100, help='Samples to be selected per step and initial samplesize')
    parser.add_argument('-vs', '--validation_size', type=int, default=0.1, help='Size of the validation set in percentage')
    parser.add_argument('-lr', '--learning_rate', type=str, default=0.01, help='Learning rate')
    parser.add_argument('-al', '--active_learning', type=bool, default=False, help='Use active learning')
    parser.add_argument('-dr', '--directory', type=str, default='_plots', help='Directory to save the ouputs')
    parser.add_argument('-tb', '--tensorboard', type=str, default='runs', help='Directory to save the TB logs')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print the model')
    args = parser.parse_args()

    model = RunModel(args.model, args.hidden_size, args.steps, args.epochs, args.dataset_type, args.sensor,
                     args.samples_per_step, args.validation_size, args.learning_rate, args.active_learning,
                     args.directory, args.tensorboard, args.verbose)

    # Train the model
    for step in range(model.steps):
        start_step = time.time()
        model.train_model(step) # Train the model
        print('Step: {} of {}, time-taken: {:.2f} seconds'.format(step+1, model.steps, time.time() - start_step))
        model.acquisition_function(args.samples_per_step) # Select the next samples from the pool

        # Save the model after each step and read it back in for the training
        #model.save_model()
        #model.load_model()
        #model.test_model()