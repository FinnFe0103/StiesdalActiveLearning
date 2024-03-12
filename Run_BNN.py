from Models.BNN import BayesianNetwork
from data import Dataprep
import argparse
from torch.optim import Adam
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import time
import random

class RunModel:
    def __init__(self, model_name, hidden_size, rounds, epochs, dataset_type, sensor, initial_samplesize, 
                 validation_size, learning_rate, active_learning, directory, tensorboard, verbose):
        
        data = Dataprep(dataset_type, initial_samplesize, sensor)
        self.known_data, self.pool_data = data.known_data, data.pool_data


        # Data selection (maybe move into the data.py file later on)
        # if dataset_type == 'Caselist':
        #     raise ValueError('Not implemented yet') # Load the caselist
        
        # elif dataset_type.split('_')[0] == 'Generated':
        #     dataset_size = int(dataset_type.split('_')[1]) # Size of the generated dataset
        #     data = Dataprep(dataset_size, initial_samplesize) # Load the data
        #     self.x_selected, self.y_selected = data.x_selected, data.y_selected # Save the known data
        #     self.x_pool, self.y_pool = data.x_pool, data.y_pool # Save the pool data

        self.validation_size = validation_size # Size of the validation set in percentage

        # Active learning parameters
        self.active_learning = active_learning # Whether to use active learning or random sampling
        self.rounds = rounds # Number of rounds for the active learning
        self.epochs = epochs # Number of epochs per round of active learning

        # Initialize the model and optimizer
        self.model_name = model_name # Name of the model
        self.model = self.init_model(hidden_size) # Initialize the model
        self.optimizer = self.init_optimizer(learning_rate) # Initialize the optimizer

        # Configs
        self.verbose = verbose # Print outputs
        self.directory = directory # Directory to save the outputs
        self.tensorboard = tensorboard # Directory to save the TB logs

    def init_model(self, hidden_size):
        if self.model_name == 'BNN':
            model = BayesianNetwork(hidden_size)
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
    
    def train_model(self):
        # INPUT TRAIN AND VALIDATION DATA
        # OUTPUT TRAINED MODEL
        # PASS TO PREDICTION METHOD THAT RETURNS THE INDICES OF THE NEXT SAMPLES TO BE SELECTED

        # Split the pool data into train and validation sets
    

        for epoch in range(self.epochs):
            if self.verbose:
                start_epoch = time.time()
                print('Epoch: {} of {}, time-taken: {:.2f} seconds'.format(epoch+1, self.epochs, time.time() - start_epoch))

        

    def select_samples(self, n):
        selected_indices = random.sample(range(len(self.pool_dataset)), n)
        print(selected_indices)
        
        # self.known_dataset.extend(random_samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the BNN model')
    parser.add_argument('-m', '--model', type=str, default='BNN', help='1. BNN, 2. MC Dropout, 3. Deep Ensembles')
    parser.add_argument('-hs', '--hidden_size', type=int, default=4, help='Number of hidden units')
    parser.add_argument('-r', '--rounds', type=int, default=2, help='Number of rounds')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('-ds', '--dataset_type', type=str, default='Caselist', help='1. Generated_2000, 2. Caselist')
    parser.add_argument('-s', '--sensor', type=str, default='foundation_origin xy FloaterOffset [m]', help='Sensor to be predicted')
    parser.add_argument('-is', '--initial_samplesize', type=int, default=100, help='Initial sample size')
    parser.add_argument('-vs', '--validation_size', type=int, default=0.1, help='Size of the validation set in percentage')
    parser.add_argument('-lr', '--learning_rate', type=str, default=0.01, help='Learning rate')
    parser.add_argument('-al', '--active_learning', type=bool, default=False, help='Use active learning')
    parser.add_argument('-dr', '--directory', type=str, default='_plots', help='Directory to save the ouputs')
    parser.add_argument('-tb', '--tensorboard', type=str, default='runs', help='Directory to save the TB logs')
    parser.add_argument('-v', '--verbose', type=bool, default=False, help='Print the model')
    args = parser.parse_args()

    model = RunModel(args.model, args.hidden_size, args.rounds, args.epochs, args.dataset_type, args.sensor,
                     args.initial_samplesize, args.validation_size, args.learning_rate, args.active_learning,
                     args.directory, args.tensorboard, args.verbose)

    print(len(model.pool_dataset))#.dataset.tensors[0].shape)

    # Train the model
    for round in range(model.rounds):
        
    



        start_round = time.time()
        model.train_model() # Train the model
        print('Round: {} of {}, time-taken: {:.2f} seconds'.format(round+1, model.rounds, time.time() - start_round))
        model.select_samples(10) # Select the next samples from the pool

        # Save the model after each round and read it back in for the training
        #model.save_model()
        #model.load_model()
        #model.test_model()