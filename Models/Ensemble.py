import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_number, use_dropout=False):
        super().__init__()

        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(p=0.5)
        self.activation = nn.Tanh()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_size))

        for _ in range(1, layer_number - 1): # Add hidden layers based on layer_number
            self.layers.append(nn.Linear(hidden_size, hidden_size))

        self.layers.append(nn.Linear(hidden_size, 1)) # Output Layer

        # # dynamically define architecture
        # self.layer_sizes = [input_dim] + n_hidden_layers * [hidden_dim] + [output_dim]
        # layer_list = [nn.Linear(self.layer_sizes[idx - 1], self.layer_sizes[idx]) for idx in
        #               range(1, len(self.layer_sizes))]
        # self.layers = nn.ModuleList(layer_list)

    def forward(self, input):
        hidden = self.activation(self.layers[0](input))
        for layer in self.layers[1:-1]:
            hidden_temp = self.activation(layer(hidden))

            if self.use_dropout:
                hidden_temp = self.dropout(hidden_temp)

            hidden = hidden_temp + hidden  # residual connection

        output_mean = self.layers[-1](hidden).squeeze()
        return output_mean