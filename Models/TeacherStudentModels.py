import torch.nn as nn
import torch.nn.functional as F
from Layers import layers


class MLPTeacher(nn.Module):
    def __init__(self, hidden_units, in_units, out_units):
        super().__init__()
        self.hidden_layer1 = layers.Linear(in_units, hidden_units,
                                           layer_id=1)  # only want to prune layer with id 1 (first hl)
        self.hidden_layer2 = layers.Linear(hidden_units, out_units)
        self.activations = []

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = F.relu(x)
        self.activations.append(x.detach().clone())
        x = self.hidden_layer2(x)
        x = F.log_softmax(x, dim=1)
        return x

    def reset_activations(self):
        self.activations = []


class MLPStudent(nn.Module):
    def __init__(self, hidden_units, in_units, out_units):
        super().__init__()
        self.hidden_layer1 = layers.Linear(in_units, hidden_units,
                                           layer_id=1)  # only want to prune layer with id 1 (first hl)
        self.hidden_layer2 = layers.Linear(hidden_units, out_units)
        self.activations = []

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = F.relu(x)
        self.activations.append(x.detach().clone())
        x = self.hidden_layer2(x)
        x = F.softmax(x, dim=1)
        return x

    def reset_activations(self):
        self.activations = []
