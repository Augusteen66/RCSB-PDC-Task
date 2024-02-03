import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class SingleNeuronModel(nn.Module):
    def __init__(self, input_size):
        super(SingleNeuronModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x)).squeeze()
    
class FullyConnectedModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FullyConnectedModel, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.relu(self.fc(x))
        return torch.sigmoid(self.output(x)).squeeze()