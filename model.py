import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        super(NeuralNetwork, self).__init__() 
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))

        # Add hidden layers
        for _ in range(hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x): 
        for layer in self.layers[:-1]: # All layers except the last one
            x = torch.relu(layer(x))
        x = self.layers[-1](x) # Output layer
        return x


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, n_hidden=128, n_layers=2):
        self.n_layers=n_layers
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        for i in range(n_layers - 1):
            setattr(self, f"layer{i+2}", nn.Linear(128, 128))
        setattr(self, f"layer{n_layers+1}", nn.Linear(128, n_actions))

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        for i in range(2, self.n_layers + 1):
            x = F.relu(getattr(self, f"layer{i}")(x))
        
        
