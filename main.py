import pandas as pd
import os

from data import Dataprep
from Models.BNN import BayesianNetwork, Train_BNN
#from train import load_data, train_model

import torch

if __name__ == '__main__':

    os.makedirs("_plots", exist_ok=True)

    data = Dataprep(-10, 10, 2000)
    train_loader, train_dataset, test_loader, test_dataset = data.train_loader, data.train_dataset, data.test_loader, data.test_dataset

    # Initialize the BNN
    bnn = BayesianNetwork()
    print(bnn)
    print("Params:", sum(p.numel() for p in bnn.parameters() if p.requires_grad))

    # Train the BNN
    #bnn.train_model(train_loader, train_dataset, test_loader, test_dataset, epochs=100)
    trainer = Train_BNN(bnn, train_loader, train_dataset, test_loader, test_dataset)
    trainer.train_model(epochs=100)

# select the next samples from the pool
# save the model after each round and read it back in for the training
