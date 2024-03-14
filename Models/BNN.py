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
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        hidden_size = hidden_size

        self.blinear1 = BayesianLinear(input_dim, hidden_size, prior_sigma_1=100) #5
        self.blinear2 = BayesianLinear(hidden_size, hidden_size, prior_sigma_1=100) #5
        self.blinear3 = BayesianLinear(hidden_size, 1, prior_sigma_1=100) #5

    def forward(self, x):
        x = torch.relu(self.blinear1(x))
        x = torch.relu(self.blinear2(x))
        return self.blinear3(x)