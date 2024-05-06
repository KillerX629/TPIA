import torch
import torch.nn as nn
import torch.optim as optim



class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers, output_dim):
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
