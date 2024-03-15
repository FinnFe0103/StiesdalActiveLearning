import torch
import torch.nn as nn
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
    
@variational_estimator
class BayesianNetwork(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_number, prior_sigma):
        super().__init__()
        
        self.layers = nn.ModuleList()  # A list to hold all layers
        self.layers.append(BayesianLinear(input_dim, hidden_size, prior_sigma_1=prior_sigma)) # Input layer 
        
        for _ in range(1, layer_number - 1):  # Add hidden layers based on layer_number
            self.layers.append(BayesianLinear(hidden_size, hidden_size, prior_sigma_1=prior_sigma))
        
        self.layers.append(BayesianLinear(hidden_size, 1, prior_sigma_1=prior_sigma))  # Output Layer

    def forward(self, x):
        for layer in self.layers[:-1]: # Apply all layers except for the last with ReLU activation
            x = torch.relu(layer(x))
        x = self.layers[-1](x) # No activation function for the last layer in this case
        return x