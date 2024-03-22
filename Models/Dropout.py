import torch.nn as nn

class Dropout(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_number):
        super().__init__()

        self.dropout = nn.Dropout(p=0.5)
        self.activation = nn.Tanh()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_size))

        for _ in range(1, layer_number - 1): # Add hidden layers based on layer_number
            self.layers.append(nn.Linear(hidden_size, hidden_size))

        self.layers.append(nn.Linear(hidden_size, 1)) # Output Layer

    def forward(self, input):
        hidden = self.activation(self.layers[0](input))
        for layer in self.layers[1:-1]:
            hidden_temp = self.activation(layer(hidden))
            hidden_temp = self.dropout(hidden_temp)
            hidden = hidden_temp + hidden  # residual connection

        output_mean = self.layers[-1](hidden).squeeze()
        return output_mean