import os
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
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
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from torch.utils.tensorboard import SummaryWriter
# Module imports
from data import Dataprep, load_data
from Models.BNN import BayesianNetwork
from Models.ExactGP import ExactGPModel
from Models.Ensemble import Ensemble
from Models.Dropout import Dropout

class RunModel:
    def __init__(self, model_name, run_name, dataset_type='Caselist', scaling=None, sensor='foundation_origin xy FloaterOffset [m]',
                learning_rate=None,
                prior_sigma=None, complexity_weight=None,
                hidden_size=None, layer_number=None,
                ensemble_size=None, 
                dropout_rate=None,
                kernel=None, lengthscale_prior=None, lengthscale_sigma=None, lengthscale_mean=None, noise_prior=None, noise_sigma=None, noise_mean=None, noise_constraint=None, lengthscale_type=None,
                acquisition_function=None, lambda_reg=None, steps=None, epochs=None, samples_per_step=33, sampling_method='Random', validation_size=0.0, topk=33,
                verbose=False):

        print("All received arguments:", locals())

        # Configs
        self.run_name = run_name # Name of the run
        self.verbose = verbose # Print outputs
        os.makedirs('_plots', exist_ok=True) # Directory to save the outputs
        current_time = datetime.datetime.now().strftime("%H%M%S") # Unique directory based on datetime for each run
        log_dir = os.path.join('/runs', current_time + '_' + model_name) # fix run_name
        self.writer = SummaryWriter(log_dir) # TensorBoard
        print('Run saved under:', log_dir) 

        # Data parameters
        self.data = Dataprep(dataset_type, sensor, scaling=scaling, initial_samplesize=samples_per_step, sampling_method=sampling_method)
        self.data_known, self.data_pool = self.data.data_known, self.data.data_pool

        # Active learning parameters
        self.active_learning = acquisition_function # Which acquisition function to use
        self.lambda_reg = lambda_reg # Regularization parameter for the acquisition function
        self.steps = steps # Number of steps for the active learning
        self.epochs = epochs # Number of epochs per step of active learning
        self.samples_per_step = samples_per_step # Number of samples to select per step
        self.validation_size = validation_size # Size of the validation set in percentage
        self.topk = topk # Number of top samples to select from the pool

        # Initialize the model and optimizer
        self.learning_rate = learning_rate # Learning rate for the optimizer
        self.model_name = model_name # Name of the model
        self.hidden_size = hidden_size
        self.layer_number = layer_number
        self.prior_sigma = prior_sigma
        self.complexity_weight = complexity_weight
        self.ensemble_size = ensemble_size
        self.dropout_rate = dropout_rate
        self.init_model(self.data_known.shape[1]-1) # Initialize the model
        self.device = torch.device('mps' if torch.backends.mps.is_available() and self.model_name != 'GP' else 'cpu') #and self.model_name != 'DE' 
        print(f'Using {self.device} for training')
        
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
        self.set_actual_pdf() # Set the actual PDF of the data


    def set_actual_pdf(self):
        y_total = np.concatenate((self.data_pool[:, -1], self.data_known[:, -1]), axis=0)
        self.y_total_actual = y_total # [observations, ]

        # Calculate the actual PDF of the data
        kde_pdf = self.calc_kde_pdf(y_total)
        self.pdf_p = kde_pdf
        return
    
    def set_predicted_pdf(self):
        # Calculate the predicted PDF of the data
        kde_pdf = self.calc_kde_pdf(self.y_total_predicted)
        self.pdf_q = kde_pdf
        return
    
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
    
    def init_noise(self, noise_prior, noise_sigma=None, noise_mean=None, noise_constraint=1e-6): # 0.4 Initialize the noise
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
    
    def train_model(self, step): # 1. Train the model
        # Split the pool data into train and validation sets
        if self.validation_size > 0:
            train, val = train_test_split(self.data_known, test_size=self.validation_size)
            if self.model_name == 'GP' or self.model_name == 'SVR': # no train and val loader needed for GP/SVR, since not operating in batches
                train_loader = train 
                val_loader = val
            elif self.model_name == 'BNN' or self.model_name == 'DE' or self.model_name == 'MCD':
                train_loader = load_data(train)
                val_loader = load_data(val)
        elif self.validation_size == 0:
            if self.model_name == 'GP' or self.model_name == 'SVR': # no train loader needed for GP/SVR, since not operating in batches
                train_loader = self.data_known 
            elif self.model_name == 'BNN' or self.model_name == 'DE' or self.model_name == 'MCD':
                train_loader = load_data(self.data_known)
        else:
            raise ValueError('Invalid validation size')

        if self.model_name == 'BNN': # BNN Training
            self.model.to(self.device) # move the model to the configured device
            for epoch in range(self.epochs):
                start_epoch = time.time()
                self.model.train()
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
                if self.verbose:
                    print(f'Step: {step+1} | Epoch: {epoch+1} of {self.epochs} | Train-Loss: {epoch_train_loss:.4f} | {time.time() - start_epoch:.2f} seconds')

            if self.validation_size > 0:
                self.evaluate_val_data(val_loader, step) # Evaluate the model on the validation set

        elif self.model_name == 'GP': # GP Training
            X_train = torch.tensor(train_loader[:, :-1]).to(self.device)
            y_train = torch.tensor(train_loader[:, -1]).to(self.device)

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
                    if self.verbose:
                        print(f'Step: {step+1} | Epoch: {epoch+1} of {self.epochs} | Train-Loss: {train_loss:.4f}  | Noise: {self.likelihood.noise.item():.3f} | {time.time() - start_epoch:.2f} seconds')
            except linear_operator.utils.errors.NotPSDError:
                print("Warning: The matrix is not positive semi-definite. Exiting this run.")
                sys.exit()
                
            if self.validation_size > 0:
                # Evaluation step here, assuming evaluate_model method exists
                self.evaluate_val_data(val_loader, step)

        elif self.model_name == 'SVR': # SVR Training
            self.model.fit(train_loader[:, :-1], train_loader[:, -1])

            if self.validation_size > 0:
                self.evaluate_val_data(val_loader, step)

        elif self.model_name == 'DE': # DE Training
            model_epoch_loss = []
            for model_idx in range(len(self.model)):
                self.model[model_idx].to(self.device)
                epoch_loss = []
                for epoch in range(self.epochs):
                    self.model[model_idx].train()
                    
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
            if self.verbose:
                print(f'Step: {step+1} | Train-Loss: {train_loss:.4f}')

            if self.validation_size > 0:
                    self.evaluate_val_data(val_loader, step)

        elif self.model_name == 'MCD': # MCD Training
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
                if self.verbose:
                    print(f'Step: {step+1} | Epoch: {epoch+1} of {self.epochs} | Train-Loss: {epoch_train_loss:.4f} | {time.time() - start_epoch:.2f} seconds')

            if self.validation_size > 0:
                self.evaluate_val_data(val_loader, step) # Evaluate the model on the validation set

    def evaluate_val_data(self, val_loader, step): # 1.1 Evaluate the model on the validation set
        if self.model_name == 'BNN': # BNN-specific evaluation process
            self.model.eval()  # Set the model to evaluation mode
            total_val_loss = 0
            with torch.no_grad():  # Inference mode, gradients not required
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    # Calculate loss using sample_elbo for Bayesian inference
                    loss = self.model.sample_elbo(inputs=x, labels=y,
                                                criterion=self.mse, sample_nbr=3,
                                                complexity_cost_weight=self.complexity_weight)
                    total_val_loss += loss.item()

            step_val_loss = total_val_loss / len(val_loader) # Average loss over batches
            self.writer.add_scalar('loss/val', step_val_loss, step+1)
            if self.verbose:
                print(f'Step: {step+1} | Val-Loss: {step_val_loss:.4f}')
        
        elif self.model_name == 'GP': # GP-specific evaluation process
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
        
        elif self.model_name == 'SVR': # SVR-specific evaluation process
            y_val = val_loader[:, -1]
            x_val = val_loader[:, :-1]
            y_pred = self.model.predict(x_val)
            loss = self.mse(y_pred, y_val)
            print('validation MSE1:', loss)

            self.writer.add_scalar('loss/val', loss, step+1)
            if self.verbose:
                print(f'Step: {step+1} | Val-Loss: {loss:.4f}')
        
        elif self.model_name == 'DE': # DE-specific evaluation process
            total_val_loss = 0
            for model in self.model:
                model.eval()
                model_val_loss = 0
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(self.device), y.to(self.device)
                        loss = self.mse(model(x), y.squeeze())
                        model_val_loss += loss.item()

                model_step_val_loss = model_val_loss / len(val_loader) # Average loss over batches for one model
                #print(f'Model Val-Loss: {model_step_val_loss:.4f}')
                total_val_loss += model_step_val_loss
            
            step_val_loss = total_val_loss / len(self.model) # Average loss over models
            self.writer.add_scalar('loss/val', step_val_loss, step+1)
            if self.verbose:
                print(f'Step: {step+1} | Val-Loss: {step_val_loss:.4f}')

        elif self.model_name == 'MCD': # MCD-specific evaluation process
            #self.model.eval() # DO NOT PUT IN EVALUATION MODE; OTHERWISE DROPOUT WILL NOT WORK
            total_val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    loss = self.mse(self.model(x), y.squeeze())
                    total_val_loss += loss.item()

            step_val_loss = total_val_loss / len(val_loader) # Average loss over batches
            self.writer.add_scalar('loss/val', step_val_loss, step+1)
            if self.verbose:
                print(f'Step: {step+1} | Val-Loss: {step_val_loss:.4f}')

    def evaluate_pool_data(self, step): # 2. Evaluate the model on the pool data
        # Convert pool data to PyTorch tensors and move them to the correct device
        x_pool_torch = torch.tensor(self.data_pool[:, :-1]).to(self.device)
        y_pool_torch = torch.tensor(self.data_pool[:, -1]).to(self.device) # [observations, ]

        if self.model_name == 'BNN':
            self.model.eval()  # Set the model to evaluation mode
            with torch.no_grad():  # No need to calculate gradients
                predictions = self.model(x_pool_torch).squeeze() # [observations, ]
                mse = self.mse(predictions, y_pool_torch)
                mae = self.mae(predictions, y_pool_torch)
                
        elif self.model_name == 'GP':
            self.model.eval()  # Set the model to evaluation mode
            with torch.no_grad():  # No need to calculate gradients
                self.likelihood.eval()  # Also set the likelihood to evaluation mode
                prediction = self.model(x_pool_torch) # [observations, ]
                predictions = self.likelihood(prediction)  # [observations, ]
                mse = self.mse(predictions.mean, y_pool_torch) 
                mae = self.mae(predictions.mean, y_pool_torch)

        elif self.model_name == 'SVR':
            predictions = self.model.predict(self.data_pool[:, :-1])
            print('EVALUATE POOL SHAPES', predictions.shape, self.data_pool[:, -1].shape)
            mse = self.mse(predictions, self.data_pool[:, -1])
            mae = self.mae(predictions, self.data_pool[:, -1])

        elif self.model_name == 'DE':
            for model in self.model:
                model.eval()
            with torch.no_grad():
                predictions = [model(x_pool_torch).clone().detach().cpu().numpy() for model in self.model]
                predictions = np.mean(predictions, axis=0)
                mse = self.mse(torch.tensor(predictions).to(self.device), y_pool_torch.squeeze())
                mae = self.mae(torch.tensor(predictions).to(self.device), y_pool_torch.squeeze())

        elif self.model_name == 'MCD':
            #self.model.eval()  # DO NOT PUT IN EVALUATION MODE; OTHERWISE DROPOUT WILL NOT WORK
            with torch.no_grad():
                predictions = self.model(x_pool_torch)
                mse = self.mse(predictions, y_pool_torch.squeeze())
                mae = self.mae(predictions, y_pool_torch.squeeze())
            
        # Log the loss to TensorBoard
        self.writer.add_scalar('loss/pool_mse', mse.item(), step + 1)
        self.writer.add_scalar('loss/pool_mae', mae.item(), step + 1)
        
        if self.verbose:
            print(f'Step: {step + 1} | Pool-Loss: {mse.item()}')
    
    def predict(self, samples=500): # 3. Predict the mean and std (uncertainty) on the pool data
        x_pool = self.data_pool[:, :-1]
        x_pool_torch = torch.tensor(x_pool).to(self.device)

        if self.model_name == 'BNN':
            self.model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                preds = [self.model(x_pool_torch) for _ in range(samples)]
            preds = torch.stack(preds)  # Shape: [samples, N, output_dim]
            means = preds.mean(dim=0).squeeze().detach().cpu().numpy()  # calculate the mean of the predictions
            stds = preds.std(dim=0).squeeze().detach().cpu().numpy()  # calculate the standard deviation of the predictions
        elif self.model_name == 'GP':
            self.model.eval()  # Set model to evaluation mode
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                self.likelihood.eval()  # Also set the likelihood to evaluation mode for GP predictions
                preds = self.model(x_pool_torch)
                observed_pred = self.likelihood(preds)
            means = observed_pred.mean.numpy()
            stds = observed_pred.stddev.numpy()
        elif self.model_name == 'SVR':
            means = self.model.predict(x_pool)
            stds = np.zeros_like(means)
        elif self.model_name == 'DE':
            for model in self.model:
                model.eval()
            with torch.no_grad():
                preds = [model(x_pool_torch).clone().detach().cpu().numpy() for model in self.model]
                means = np.mean(preds, axis=0)
                stds = np.std(preds, axis=0)
            print('MEANS, STDS:', means.shape, stds.shape)
        elif self.model_name == 'MCD':
            #self.model.eval() # DO NOT PUT IN EVALUATION MODE; OTHERWISE DROPOUT WILL NOT WORK
            with torch.no_grad():
                preds = [self.model(x_pool_torch) for _ in range(samples)]
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

    def acquisition_function(self, means, stds): # 4. Select the next samples from the pool
        stds = stds.copy()
        means = means.copy()

        if self.active_learning == 'RS': # Random Sampling
            selected_indices = random.sample(range(len(self.data_pool)), self.samples_per_step)
            return selected_indices
        elif self.active_learning == 'EX':  # Exploitation: next sample highest predictions
            selected_indices = np.argsort(means)[-self.samples_per_step:]
            return selected_indices
        
        best_y = np.max(self.data_known[:, -1])
        distance_matrix = cdist(self.data_pool, self.data_pool, 'euclidean') # Distance matrix between data points
        
        selected_indices_this_step = []
        counter = 0
        for _ in range(self.samples_per_step):
            if selected_indices_this_step:
                current_min_distances = distance_matrix[:, selected_indices_this_step].min(axis=1)
            else:
                current_min_distances = np.zeros(len(self.data_pool))  # No diversity penalty if no points have been selected yet

            if self.active_learning == 'US':  # Uncertainty Sampling
                scores = stds + self.lambda_reg * current_min_distances
            elif self.active_learning == 'EI':  # Expected Improvement
                z = (means - best_y) / stds
                scores = (means - best_y) * norm.cdf(z) + stds * norm.pdf(z) + self.lambda_reg * current_min_distances
            elif self.active_learning == 'PI':  # Probability of Improvement
                z = (means - best_y) / stds
                scores = norm.cdf(z) + self.lambda_reg * current_min_distances
            elif self.active_learning == 'UCB':  # Upper Confidence Bound
                scores = means + 2.0 * stds + self.lambda_reg * current_min_distances
            else:
                raise ValueError('Invalid acquisition function')

            # Select the index with the highest score, accounting for diversity
            selected_index = scores.argsort()[-1]
            selected_indices_this_step.append(selected_index)

            # For this example, set selected stds and means to -inf to not select them again
            stds[selected_index] = -np.inf
            means[selected_index] = -np.inf
            counter += 1

        return selected_indices_this_step
    
    def plot(self, means, stds, selected_indices, step, x_highest_pred_n, y_highest_pred_n, x_highest_actual_n, y_highest_actual_n, x_highest_actual_1, y_highest_actual_1, kl_divergence_for_plot, x_points, p, q, y_min_extended, y_max_extended): #4.1 Plot the predictions and selected indices
        x_pool = self.data_pool[:, :-1] # [observations, features]
        y_pool = self.data_pool[:, -1] # [observations]
        x_selected = self.data_known[:, :-1] # [observations, features]
        y_selected = self.data_known[:, -1] # [observations]
        pca_applied = False
        # Check if dimensionality reduction is needed
        if x_pool.shape[1] > 1:
            pca = PCA(n_components=1)
            x_pool = pca.fit_transform(x_pool) # [observations, 1]
            x_selected = pca.transform(x_selected) # [observations, 1]
            x_highest_pred_n = pca.transform(x_highest_pred_n) # [observations, 1]
            x_highest_actual_n = pca.transform(x_highest_actual_n) # [observations, 1]
            x_highest_actual_1 = pca.transform(x_highest_actual_1.reshape(1, -1)) # [observations, 1]
            #print('Explained variance by the first princiapal components:', pca.explained_variance_ratio_)
            pca_applied = True
            
        x_pool_selected = x_pool[selected_indices] # [observations, 1]
        y_pool_selected = y_pool[selected_indices] # [observations]

        y_vals = [means, means + 2 * stds, means - 2 * stds] #list of 3 arrays of shape [observations in pool, 1] (use 2 for 95% CI)
        df = pd.concat([pd.DataFrame({'x': x_pool.squeeze(), 'y': y_val.squeeze()}) for y_val in y_vals], ignore_index=True)

        # Plotting
        fig = plt.figure(figsize=(8, 6), dpi=120)
        sns.lineplot(data=df, x="x", y="y", alpha=0.2)
        plt.scatter(x_pool, y_pool, c="green", marker="*", alpha=0.1)  # Plot the data pairs in the pool
        plt.scatter(x_selected, y_selected, c="red", marker="*", alpha=0.1)  # plot the train data on top
        plt.scatter(x_pool_selected, y_pool_selected, c="blue", marker="o", alpha=0.8)  # Highlight selected data points
        plt.scatter(x_highest_pred_n, y_highest_pred_n, c="purple", marker="o", alpha=0.8)
        plt.scatter(x_highest_actual_n, y_highest_actual_n, c="orange", marker="o", alpha=0.1)
        plt.scatter(x_highest_actual_1, y_highest_actual_1, c="red", marker="o", alpha=0.1)
        plt.title(self.run_name.replace("_", " ") + f' | Step {step + 1}', fontsize=6)
        plt.xlabel('1 Principal Component' if pca_applied else 'x')
        plt.legend(['Mean prediction', 'Confidence Interval', 'Pool data (unseen)', 'Seen data', 'Selected data', 'Final Prediction', 'Highest Actual'], fontsize=8)
        plt.close(fig)

        # Log the table figure
        self.writer.add_figure(f'Prediction vs Actual Table Epoch {step + 1}', fig, step + 1)
        
        if self.data_pool[:, :-1].shape[1] > 1:
            pca = PCA(n_components=2)
            data_plot = pca.fit_transform(self.data_pool[:, :-1]) # [observations, 2]
            # Plotting
            fig = plt.figure(figsize=(8, 6), dpi=120)
            plt.scatter(data_plot[:, 0], data_plot[:, 1], c='lightgray', s=30, label='Data points')
            plt.scatter(data_plot[selected_indices, 0], data_plot[selected_indices, 1], c='blue', s=50, label='Selected Samples')
            plt.title(self.run_name.replace("_", " ") + f' | Step {step + 1}', fontsize=10)
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.legend(fontsize='small')
            plt.close(fig)
            
            self.writer.add_figure(f'First two PCs with selected datapoints {step + 1}', fig, step + 1)


        # PDFs and KL divergence
        fig, axs = plt.subplots(1, 2, figsize=(18, 6))

        axs[0].plot(x_points, p, color='blue', label='Actual')
        axs[0].plot(x_points, q, color='red', label='Predicted')
        axs[0].legend(loc='upper right')
        axs[0].set_title('PDFs of Actual and Predicted Data')
        axs[0].set_xlabel('Data Points (Predicted and Actual Output)')
        axs[0].set_ylabel('(Log-) Probability Density')

        # Plot the KL divergence
        axs[1].plot(x_points, kl_divergence_for_plot, color='purple')
        axs[1].fill_between(x_points, kl_divergence_for_plot, color='purple', alpha=0.1)
        axs[1].set_title('Pointwise KL Divergence')
        axs[1].set_xlabel('Data Points (Predicted and Actual Output)')
        axs[1].set_ylabel('Pointwise KL Divergence')

        plt.tight_layout()
        plt.close(fig)
        self.writer.add_figure(f'PDFs and KL Divergence {step + 1}', fig, step + 1)

    def update_data(self, selected_indices): # 5. Update the known and pool data
        self.data_known = np.append(self.data_known, self.data_pool[selected_indices], axis=0)
        self.data_pool = np.delete(self.data_pool, selected_indices, axis=0)

    def final_prediction(self, step, samples=100):
        x_total = np.concatenate((self.data_pool[:, :-1], self.data_known[:, :-1]), axis=0)
        y_total = np.concatenate((self.data_pool[:, -1], self.data_known[:, -1]), axis=0) # [observations, ]

        x_total_torch = torch.tensor(x_total).to(self.device)
        y_total_torch = torch.tensor(y_total).to(self.device)  # y pool torch: [observations, ]
        
        if self.model_name == 'BNN':
            self.model.eval()
            with torch.no_grad():
                preds = [self.model(x_total_torch) for _ in range(samples)]
            preds = torch.stack(preds) # [samples, observations, 1]
            means = preds.mean(dim=0).detach().cpu().numpy().squeeze() # [observations, ]
            mse = self.mse(torch.tensor(means), y_total_torch)
            mae = self.mae(torch.tensor(means), y_total_torch)
        elif self.model_name == 'GP':
            self.model.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                self.likelihood.eval()  # Also set the likelihood to evaluation mode for GP predictions
                try:
                    pred = self.model(x_total_torch)
                    preds = self.likelihood(pred)
                    means = preds.mean.numpy()
                    mse = self.mse(preds.mean, y_total_torch) # y total torch: [observations, ]
                    mae = self.mae(preds.mean, y_total_torch) # y total torch: [observations, ]
                except linear_operator.utils.errors.NotPSDError:
                    print("Warning: The matrix is not positive semi-definite. Exiting this run.")
                    sys.exit()
        elif self.model_name == 'SVR':
            means = self.model.predict(x_total)
            mse = self.mse(means, y_total)
            mae = self.mae(means, y_total)
        elif self.model_name == 'DE':
            for model in self.model:
                model.eval()
            with torch.no_grad():
                preds = [model(x_total_torch).clone().detach().cpu().numpy() for model in self.model]
            means = np.mean(preds, axis=0)
            print('EVALUATE TOTAL SHAPES', means.shape, y_total.shape)
            mse = self.mse(torch.tensor(means).to(self.device), y_total_torch.squeeze())
            mae = self.mae(torch.tensor(means).to(self.device), y_total_torch.squeeze())
        elif self.model_name == 'MCD':
            #self.model.eval() # DO NOT PUT IN EVALUATION MODE; OTHERWISE DROPOUT WILL NOT WORK
            with torch.no_grad():
                preds = [self.model(x_total_torch) for _ in range(samples)]
            preds = torch.stack(preds)
            means = preds.mean(dim=0).detach().cpu().numpy()
            print('EVALUATE TOTAL SHAPES', means.shape, y_total.shape)
            mse = self.mse(torch.tensor(means).to(self.device), y_total_torch.squeeze())
            mae = self.mae(torch.tensor(means).to(self.device), y_total_torch.squeeze())
            
        highest_indices_pred_n = np.argsort(means)[-self.topk:]
        highest_indices_pred_1 = np.argsort(means)[-1]
        highest_indices_actual_n = np.argsort(y_total)[-self.topk:]
        highest_indices_actual_1 = np.argsort(y_total)[-1]

        x_highest_pred_n = x_total[highest_indices_pred_n]
        y_highest_pred_n = y_total[highest_indices_pred_n]

        x_highest_actual_n = x_total[highest_indices_actual_n]
        y_highest_actual_n = y_total[highest_indices_actual_n]

        x_highest_actual_1 = x_total[highest_indices_actual_1]
        y_highest_actual_1 = y_total[highest_indices_actual_1]

        common_indices = np.intersect1d(highest_indices_pred_n, highest_indices_actual_n)
        percentage_common = (len(common_indices) / len(highest_indices_pred_n)) * 100

        num_pool_data = len(self.data_pool)
        num_from_pool = np.sum(highest_indices_pred_n < num_pool_data)
        num_from_known = len(highest_indices_pred_n) - num_from_pool

        # Log the loss to TensorBoard
        self.writer.add_scalar('loss/total_mse', mse.item(), step + 1)
        self.writer.add_scalar('loss/total_mae', mae.item(), step + 1)

        highest_actual_in_top = False
        if any(np.array_equal(row, x_highest_actual_1) for row in x_highest_pred_n):
            highest_actual_in_top = True
            if self.verbose:
                print("FOUND: The highest actual value is in the top predictions")
        else:
            if self.verbose:
                print("NOT FOUND: The highest actual value is not in the top predictions")
        if self.verbose:
            print(f'Percentage of common indices in top {self.topk} predictions: {percentage_common:.2f}%')
            print(f'Number of predictions from pool: {num_from_pool} | Number of predictions from known data: {num_from_known}')

        highest_actual_in_known = False
        if any(np.array_equal(row, x_highest_actual_1) for row in self.data_known[:, :-1]):
            highest_actual_in_known = True
            if self.verbose:
                print("KNOWN: The highest actual value is in the known data")
        else:
            if self.verbose:
                print("NOT KNOWN: The highest actual value is not in the known data")
        
        self.y_total_predicted = means
        self.set_predicted_pdf()
        
        return x_highest_pred_n, y_highest_pred_n, x_highest_actual_n, y_highest_actual_n, x_highest_actual_1, y_highest_actual_1, mse.item(), mae.item(), percentage_common, highest_actual_in_top, highest_actual_in_known


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run the BNN model')

    # Dataset parameters
    parser.add_argument('-m', '--model', type=str, default='BNN', help='1. BNN, 2. GP, 3. MCD, 4. DE, 5. SVM')
    parser.add_argument('-ds', '--dataset_type', type=str, default='Caselist', help='1. Generated_2000, 2. Caselist')
    parser.add_argument('-sc', '--scaling', type=str, default='Standard', help='Scaling to be used: 1. Standard, 2. Minmax, 3. None')
    parser.add_argument('-se', '--sensor', type=str, default='foundation_origin xy FloaterOffset [m]', help='Sensor to be predicted')
    
    # Model parameters
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Learning rate')
    
    # BNN
    parser.add_argument('-ps', '--prior_sigma', type=float, default=0.0000001, help='Prior sigma')
    parser.add_argument('-cw', '--complexity_weight', type=float, default=0.01, help='Complexity weight')

    # NNs
    parser.add_argument('-hs', '--hidden_size', type=int, default=4, help='Number of hidden units')
    parser.add_argument('-ln', '--layer_number', type=int, default=3, help='Number of layers')

    # DE
    parser.add_argument('-es', '--ensemble_size', type=int, default=5, help='Number of models in the ensemble')

    # Dropout
    parser.add_argument('-dp', '--dropout', type=float, default=0.5, help='Dropout rate')

    # GP
    parser.add_argument('-kl', '--kernel', type=str, default='RBF', help='Kernel function for GP: 1. RBF, 2. Matern, 3. Linear, 4. Cosine, 5. Periodic')
    parser.add_argument('-lpr', '--lengthscale_prior', type=str, default=None, help='Set prior for Lengthscale 1. Gamma 2. Normal') 
    parser.add_argument('-sls', '--lengthscale_sigma', type=float, default=0.2, help='Lengthscale Sigma for GP kernel')
    parser.add_argument('-mls', '--lengthscale_mean', type=float, default=2.0, help='Lengthscale Mean for GP kernel')
    parser.add_argument('-npr', '--noise_prior', type=str, default=None, help='Set prior for Noise 1. Gamma 2. Normal') 
    parser.add_argument('-sns', '--noise_sigma', type=float, default=0.1, help='Noise Sigma for GP')
    parser.add_argument('-mns', '--noise_mean', type=float, default=1.1, help='Noise Mean for GP')
    parser.add_argument('-nc', '--noise_constraint', type=float, default=1e-3, help='Noise Constraint for GP')
    parser.add_argument('-tls', '--lengthscale_type', type=str, default='Single', help='Lengthscale Type for GP kernel 1. Single, 2. ARD')
    
    # Active learning parameters
    parser.add_argument('-af', '--acquisition_function', type=str, default='UCB', help='Type of active learning/acquisition function: 1. US, 2. RS, 3. EI, 4. PI, 5. UCB') # Uncertainty, Random, Expected Improvement, Probability of Improvement, Upper Confidence Bound
    parser.add_argument('-la', '--lambda_reg', type=float, default=0.01, help='Regularization parameter for uncertainty sampling')
    parser.add_argument('-s', '--steps', type=int, default=2, help='Number of steps')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('-ss', '--samples_per_step', type=int, default=33, help='Samples to be selected per step and initial samplesize')
    parser.add_argument('-sm', '--sampling_method', type=str, default='Random', help='Sampling method for initial samples. 1. Random, 2. LHC')
    parser.add_argument('-vs', '--validation_size', type=float, default=0, help='Size of the validation set in percentage')
    parser.add_argument('-t', '--topk', type=int, default=33, help='Number of top predictions to be selected')
    
    # Output parameters
    parser.add_argument('-dr', '--directory', type=str, default='_plots', help='Sub-directory to save the ouputs')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print the model')
    parser.add_argument('-plot', '--plot', action='store_true', help='Generate a plot of the output every step')
    
    args = parser.parse_args()

    if args.model == 'BNN':
        opt_list = ['-sc', '-hs', '-ln', '-ps', '-cw', '-lr', '-al', '-s', '-e', '-ss', '-vs', '-la']
    elif args.model == 'GP':
        opt_list = ['-sc', '-lr', '-al', '-s', '-e', '-ss', '-vs', '-kl', '-lpr', '-npr', '-sls', '-sns', '-mls', '-mns', '-nc', '-tls', '-la']
    elif args.model == 'SVR':
        opt_list = ['-sc', '-al', '-s', '-e', '-ss', '-vs', '-la']
    elif args.model == 'DE':
        opt_list = ['-sc', '-hs', '-ln', '-lr', '-al', '-s', '-e', '-ss', '-vs', '-es', '-la']
    elif args.model == 'MCD':
        opt_list = ['-sc', '-hs', '-ln', '-lr', '-al', '-s', '-e', '-ss', '-vs', '-dp', '-la']
    option_value_list = [(action.option_strings[0].lstrip('-'), getattr(args, action.dest)) 
                         for action in parser._actions if action.option_strings and action.option_strings[0] in opt_list]
    run_name = '_'.join(f"{abbr}{str(value)}" for abbr, value in option_value_list)
    
    model = RunModel(model_name=args.model, run_name=run_name, dataset_type=args.dataset_type, scaling=args.scaling, sensor=args.sensor, # Dataset parameters
                     learning_rate=args.learning_rate, # Model parameters
                     prior_sigma=args.prior_sigma, complexity_weight=args.complexity_weight, # BNN parameters
                     hidden_size=args.hidden_size, layer_number=args.layer_number, # NN parameters
                     ensemble_size=args.ensemble_size, # DE parameters
                     dropout_rate=args.dropout, # MCD parameters
                     kernel=args.kernel, lengthscale_prior=args.lengthscale_prior, lengthscale_sigma=args.lengthscale_sigma, lengthscale_mean=args.lengthscale_mean, noise_prior=args.noise_prior, noise_sigma=args.noise_sigma, noise_mean=args.noise_mean, noise_constraint=args.noise_constraint, lengthscale_type=args.lengthscale_type, # GP parameters
                     acquisition_function=args.acquisition_function, lambda_reg=args.lambda_reg, steps=args.steps, epochs=args.epochs, samples_per_step=args.samples_per_step, sampling_method=args.sampling_method, validation_size=args.validation_size, topk=args.topk, # Active learning parameters
                     directory=args.directory, verbose=args.verbose, plot=args.plot) # Output parameters

    # Iterate through the steps of active learning
    result = pd.DataFrame()

    for step in range(model.steps):
        start_step = time.time()

        train_time = time.time()
        model.train_model(step) # Train the model
        print(f'---Training time: {time.time() - train_time:.2f} seconds')

        fp_time = time.time()
        x_highest_pred_n, y_highest_pred_n, x_highest_actual_n, y_highest_actual_n, x_highest_actual_1, y_highest_actual_1, mse, mae, percentage_common, highest_actual_in_top, highest_actual_in_known = model.final_prediction(step) # Get the final predictions as if this was the last step
        print(f'---Final prediction time: {time.time() - fp_time:.2f} seconds')

        ev_pool_time = time.time()
        model.evaluate_pool_data(step) # Evaluate the model on the pool data
        print(f'---Evaluation on pool data time: {time.time() - ev_pool_time:.2f} seconds')

        predict_time = time.time()
        means, stds = model.predict() # Predict the uncertainty on the pool data
        print(f'---Prediction time: {time.time() - predict_time:.2f} seconds')

        af_time = time.time()
        selected_indices = model.acquisition_function(means, stds) # Select the next samples from the pool
        print(f'---Acquisition function time: {time.time() - af_time:.2f} seconds')

        # KL diveregence
        kl_time = time.time()
        kl_divergence_for_plot, x_points, p, q, y_min_extended, y_max_extended, kl_divergence = model.get_kl_divergence()
        print(f'---KL divergence time: {time.time() - kl_time:.2f} seconds')

        if args.plot:
            plot_time = time.time()
            model.plot(means, stds, selected_indices, step, x_highest_pred_n, y_highest_pred_n, x_highest_actual_n, y_highest_actual_n, x_highest_actual_1, y_highest_actual_1, kl_divergence_for_plot, x_points, p, q, y_min_extended, y_max_extended) # Plot the predictions and selected indices
            print(f'---Plotting time: {time.time() - plot_time:.2f} seconds')

        update_time = time.time()
        model.update_data(selected_indices) # Update the known and pool data
        print(f'Updated pool and known data:', model.data_pool.shape, model.data_known.shape)
        print(f'---Update data time: {time.time() - update_time:.2f} seconds')

        step_results = pd.DataFrame({'Step': [step+1], 'MSE': [mse], 'MAE': [mae], 'Selected Indices': [selected_indices], 'Percentage Common': [percentage_common], 'Highest Actual Value in Top Predictions': [highest_actual_in_top], 'Highest Actual Value in Knonw Data': [highest_actual_in_known], 'KL-Divergence': [kl_divergence]})
        result = pd.concat([result, pd.DataFrame({'Step': [step+1], 'MSE': [mse], 'MAE': [mae], 'Selected Indices': [selected_indices], 'Percentage Common': [percentage_common], 'Highest Actual Value in Top Predictions': [highest_actual_in_top], 'Highest ACtual Value in Knonw Data': [highest_actual_in_known], 'KL-Divergence': [kl_divergence]})], ignore_index=True)

        print(f'Step: {step+1} of {model.steps} | {time.time() - start_step:.2f} seconds')
        print('--------------------------------')

    result.to_excel(f'{args.model}_{run_name}.xlsx', index=False)
    print(f'Results saved to Excel file: {args.model}_{run_name}')

    # BNN: -v -m BNN -ln 3 -hs 4 -ps 0.0000001 -cw 0.001 -al UCB -s 5 -e 100
    # DE: -v -m DE -es 2 -al UCB
    # MCD: -v -m MCD -hs 20 -ln 5 -dp 0.5 -al UCB -s 5
    # GP:
    # -v -m GP -sc Minmax -lr 0.1 -al US -s 10 -e 150 -ss 25 -vs 0.2 -kl Matern
    # -v -m GP -sc Minmax -lr 0.1 -al US -s 10 -e 150 -ss 25 -vs 0.2 -kl RBF
    # -v -m GP -sc Minmax -lr 0.1 -al US -s 10 -e 150 -ss 25 -vs 0.2 -kl Linear
    # -v -m GP -sc Minmax -lr 0.1 -al US -s 10 -e 150 -ss 25 -vs 0.2 -kl Cosine
    # -v -m GP -sc Minmax -lr 0.1 -al US -s 10 -e 150 -ss 25 -vs 0.2 -kl Periodic
    # -v -m GP -sc Minmax -lr 0.1 -al US -s 10 -e 150 -ss 25 -vs 0.2 -kl "RBF+Linear"
    # -v -m GP -sc Minmax -lr 0.1 -al US -s 10 -e 150 -ss 25 -vs 0.2 -kl "RBF+Cosine"
    # -v -m SVR -al EX -s 10 -e 150 -ss 25