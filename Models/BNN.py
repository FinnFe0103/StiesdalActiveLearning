import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
from torch.utils.tensorboard import SummaryWriter

@variational_estimator
class BayesianNetwork(nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        hidden_size = hidden_size

        self.blinear1 = BayesianLinear(1, hidden_size, prior_sigma_1=50)
        self.blinear2 = BayesianLinear(hidden_size, hidden_size, prior_sigma_1=50)
        self.blinear3 = BayesianLinear(hidden_size, 1, prior_sigma_1=50)

    def forward(self, x):
        x = torch.relu(self.blinear1(x))
        x = torch.relu(self.blinear2(x))
        return self.blinear3(x)

################################################################
################################################################
#### OLD CODE - yet to be implemented into the RUN_BNN code ####
################################################################
################################################################
class Train_BNN:
    def __init__(self, model, train_loader, train_dataset, test_loader, test_dataset):
        self.model = model

        self.train_loader = train_loader
        self.train_dataset = train_dataset
        self.test_loader = test_loader
        self.test_dataset = test_dataset

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        # Unique directory based on datetime for each run
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join('Models/runs', 'BNN_' + current_time)
        self.writer = SummaryWriter(log_dir) # TensorBoard
        
    def train_model(self, epochs=100):
        self.model.to(self.device) # move the model to the configured device

        for epoch in range(1, epochs+1):
            self.model.train()
            total_loss = 0
            for batch_idx, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                loss = self.model.sample_elbo(inputs=x,
                                            labels=y,
                                            criterion=self.criterion,
                                            sample_nbr=1,
                                            complexity_cost_weight=0.01/len(self.train_dataset))
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            # Log model parameters
            for name, param in self.model.named_parameters():
                self.writer.add_histogram(f'{name}', param, epoch)
                if param.grad is not None:
                    self.writer.add_histogram(f'{name}.grad', param.grad, epoch)

            # Average loss for the current epoch
            epoch_loss = total_loss / len(self.train_loader)
            self.writer.add_scalar('Loss/train', epoch_loss, epoch)

            # After every round of active learning
            if epoch == epochs:
                test_loss = self.evaluate_model(epoch)
                self.writer.add_scalar('Loss/test', test_loss, epoch)

    def estimate_uncertainty(self, x_pool, samples=100):
        self.model.eval()
        
        with torch.no_grad():
            # Collect multiple predictions
            preds = [self.model(x_pool.to(self.device)) for _ in range(samples)]
            preds = torch.stack(preds)  # Shape: [samples, N, output_dim]
        
        # Calculate uncertainty for each sample
        uncertainties = preds.var(dim=0).mean(dim=1).cpu()  # Shape: [N], variance over samples, averaged across output dimensions
        print(uncertainties)
        return uncertainties

    def select_active_samples(self, x_pool, y_pool, num_samples):
        # Estimate uncertainty of the pool set
        uncertainties = self.estimate_uncertainty(x_pool)

        # Select indices with highest uncertainty
        _, indices = torch.topk(uncertainties, num_samples)
        return x_pool[indices], y_pool[indices], indices



    #### AFTER EVERY ROUND OF ACTIVE LEARNING ####
    def evaluate_model(self, epoch):
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0

        with torch.no_grad():  # Inference mode, gradients not required
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                # Calculate loss using sample_elbo for Bayesian inference
                loss = self.model.sample_elbo(inputs=x,
                                            labels=y,
                                            criterion=self.criterion,
                                            sample_nbr=3,
                                            complexity_cost_weight=0.01/len(self.test_dataset))
                total_loss += loss.item()

        test_loss = total_loss / len(self.test_loader)
        print(f"Epoch {epoch} | Test loss: {test_loss}")

        self.log_predictions_and_plot(epoch)
        
        return test_loss
    
    # log the predictions in a csv and the plots in tensorboard
    def log_predictions_and_plot(self, epoch, samples=500):
        x_selected = self.train_dataset.tensors[0].cpu().numpy() # x values of the 'train' dataset
        y_selected = self.train_dataset.tensors[1].cpu().numpy() # y values of the 'train' dataset

        tensor_x_pool = self.test_dataset.tensors[0].to(self.device) # x values of the 'test' dataset (in tensor)
        x_pool = tensor_x_pool.squeeze().cpu().numpy() # x values of the 'test' dataset
        y_pool = self.test_dataset.tensors[1].cpu().numpy() # y values of the 'test' dataset
        
        preds = [self.model(tensor_x_pool) for _ in range(samples)]  # make predictions (automatically calls the forward method)
        preds = torch.stack(preds) # stack the predictions
        means = preds.mean(axis=0).detach().cpu().numpy() # calculate the mean of the predictions
        stds = preds.std(axis=0).detach().cpu().numpy() # calculate the standard deviation of the predictions

        # Prepare data for CSV
        predictions_df = pd.DataFrame({
            'x': x_pool.flatten(),
            'y_actual': y_pool.flatten(),
            'y_pred_mean': means.flatten(),
            'y_pred_std': stds.flatten(),
        })
        # Save predictions to CSV
        predictions_df.to_csv(f"_plots/predictions_epoch_{epoch}.csv", index=False)

        # Prepare data for plotting
        dfs = []
        y_vals = [means, means + 2 * stds, means - 2 * stds]
        for i in range(3): #len(y_vals)
            dfs.append(pd.DataFrame({'x': x_pool, 'y': y_vals[i].squeeze()}))
        df = pd.concat(dfs)

        # Plotting
        fig = plt.figure()
        sns.lineplot(data=df, x="x", y="y")
        plt.scatter(x_pool, y_pool, c="green", marker="*", alpha=0.1)  # Plot actual y values
        plt.scatter(x_selected, y_selected, c="red", marker="*", alpha=0.2) # plot train data on top
        plt.title(f'Predictions vs Actual Epoch {epoch}')
        plt.legend(['Mean prediction', '+2 Std Dev', '-2 Std Dev', 'Actual'])
        plt.close(fig)

        # Log the table figure
        self.writer.add_figure(f'Prediction vs Actual Table Epoch {epoch}', fig, epoch)